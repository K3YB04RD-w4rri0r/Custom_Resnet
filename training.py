import argparse
import os
from typing import List, Sized
import torch
import torch.nn as nn
import torchmetrics
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from audit import check_parameter_count
from network import BasicResBlock, BottleneckBlock, CustomResNet, DEEPCNN
from pretrained import Bottleneck, ResNet
import wandb


def collate_fn(batch_list: List[tuple[torch.Tensor, int]]):
    images, labels = zip(*batch_list)
    return torch.stack(images), torch.tensor(labels)


def train_one_epoch(model: nn.Module, optimizer, loss_fn, loader: DataLoader, device: str):
    model.train()
    cumulative_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=10).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=10).to(device)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=10).to(device)
    auc = torchmetrics.AUROC(task="multiclass", num_classes=10).to(device)

    for images, labels in loader:
        images = images.to(device)
        y = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            cumulative_loss += len(labels) * loss.item()
            accuracy.update(probs, y)
            recall.update(probs, y)
            precision.update(probs, y)
            f1.update(probs, y)
            auc.update(probs, y)

    assert isinstance(loader.dataset, Sized)
    return {
        "loss": cumulative_loss / len(loader.dataset),
        "accuracy": accuracy.compute().item(),
        "recall": recall.compute().item(),
        "precision": precision.compute().item(),
        "f1": f1.compute().item(),
        "auc": auc.compute().item(),
    }


def validate(model: nn.Module, loss_fn, loader: DataLoader, device: str):
    model.eval()
    cumulative_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=10).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=10).to(device)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=10).to(device)
    auc = torchmetrics.AUROC(task="multiclass", num_classes=10).to(device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            y = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, y)

            probs = torch.softmax(logits, dim=1)
            cumulative_loss += len(labels) * loss.item()
            accuracy.update(probs, y)
            recall.update(probs, y)
            precision.update(probs, y)
            f1.update(probs, y)
            auc.update(probs, y)

    assert isinstance(loader.dataset, Sized)
    return {
        "loss": cumulative_loss / len(loader.dataset),
        "accuracy": accuracy.compute().item(),
        "recall": recall.compute().item(),
        "precision": precision.compute().item(),
        "f1": f1.compute().item(),
        "auc": auc.compute().item(),
    }


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train image classification models from configuration")
    parser.add_argument("--model", type=str, default="CustomResnet", help="Model config to use from yaml")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer name to use from yaml")
    args = parser.parse_args()

    config = load_config("config.yaml")
    run = wandb.init(project="ResNet", config=config)
    config = run.config

    device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["seed"])
    current_epoch = 0


    model_name = args.model
    models_dict = config["experiment_configs"]["models"]
    if model_name not in models_dict:
        raise KeyError(f"Selected model '{model_name}' not found in configuration file under models.")

    model_config = models_dict[model_name]
    architecture = model_config.get("architecture", model_name)


    if architecture.startswith("Resnet") or model_name.startswith("CustomResnet"):
        block_type = BasicResBlock if model_config.get("block_type") == "basic" else BottleneckBlock
        model = CustomResNet(
            in_channels=model_config["in_channels"],
            num_classes=model_config["num_classes"],
            schema=model_config["schema"],
            block_type=block_type,
        ).to(device)

    elif architecture.startswith("Deep") or model_name.startswith("DEEPCNN"):
        model = DEEPCNN(
            in_channels=model_config["in_channels"],
            inner_channels=model_config["inner_channels"],
            num_inner_blocks=model_config["num_inner_blocks"],
        ).to(device)

    elif architecture.startswith("Imported") or model_name.startswith("ImportedNet"):
        model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=10).to(device)
    else:
        raise ValueError(f"Unrecognized model configuration: {model_name} / {architecture}")


    opt_name = args.optimizer
    opt_config = None
    for opt in config["experiment_configs"]["optimizers"]:
        if opt["name"] == opt_name:
            opt_config = opt
            break

    if opt_config is None:
        raise ValueError(f"Optimizer {opt_name} was not found in config.yaml.")

    if opt_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config["lr"],
            betas=opt_config["betas"],
        )
    else:
        raise NotImplementedError(f"Optimizer configuration for {opt_name} has not been implemented in the script.")

    run.watch(model, log="gradients", log_freq=100)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_path_dir = config["experiment_configs"]["checkpoint_path"]
    checkpoint = os.path.join(checkpoint_path_dir, f"{architecture}.pth")
    os.makedirs(checkpoint_path_dir, exist_ok=True)

    if os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        if architecture.startswith("Imported"):
            try:
                model_state_dict = checkpoint_dict["model_state"]
                model_state_dict = {
                    k: v
                    for k, v in model_state_dict.items()
                    if ((not k.startswith("fc.")) and (not k.startswith("conv1.")))
                }
                model.load_state_dict(model_state_dict, strict=False)

                for param in model.parameters():
                    param.requires_grad = False
                for param in model.conv1.parameters():
                    param.requires_grad = True
                for param in model.fc.parameters():
                    param.requires_grad = True
            except Exception as e:
                print("Error loading checkpoint: ", e)
        else:
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
            current_epoch = checkpoint_dict["checkpoint_epoch"]

    check_parameter_count(model=model)

    # Dynamic transform construction
    transform_list = []
    aug_configs = config.get("augmentations", {}).get("train", [])
    for aug in aug_configs:
        name = aug.get("name")
        if name == "ResizeCrop":
            transform_list.append(transforms.RandomResizedCrop(size=(32, 32)))
        elif name == "HorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip())
        elif name == "ColorJitter":
            transform_list.append(transforms.ColorJitter())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_train = transforms.Compose(transform_list)

    trainset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = datasets.CIFAR10(root="./data", train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    num_epochs = config["experiment_configs"]["num_epochs"]
    for epoch in range(current_epoch, num_epochs):

        if epoch % 10 == 0:
            state_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "checkpoint_epoch": epoch,
            }
            torch.save(f=checkpoint, obj=state_dict)

            artifact = wandb.Artifact(name=f"{architecture}_state", type="model", metadata={"epoch": epoch})
            artifact.add_file(checkpoint)
            run.log_artifact(artifact)
            print("Just uploaded model checkpoint to: ", checkpoint)

        print(f"Current epoch: {epoch}")
        epoch_train = train_one_epoch(
            model=model, optimizer=optimizer, loss_fn=loss_fn, loader=trainloader, device=device
        )
        epoch_val = validate(model=model, loss_fn=loss_fn, loader=testloader, device=device)
        run.log({
            "epoch": epoch,
            # train
            "train_loss": epoch_train["loss"],
            "train_accuracy": epoch_train["accuracy"],
            "train_recall": epoch_train["recall"],
            "train_precision": epoch_train["precision"],
            "train_f1": epoch_train["f1"],
            "train_auc": epoch_train["auc"],
            # val
            "val_loss": epoch_val["loss"],
            "val_accuracy": epoch_val["accuracy"],
            "val_recall": epoch_val["recall"],
            "val_precision": epoch_val["precision"],
            "val_f1": epoch_val["f1"],
            "val_auc": epoch_val["auc"],
        })


if __name__ == "__main__":
    main()