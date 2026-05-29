import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from network import DEEPCNN
from typing import List, Callable
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import os
import torchmetrics

def collate_fn(batch_list : List[tuple[torch.Tensor, int]]):
    images, labels = zip(*batch_list)
    return torch.stack(images), torch.tensor(labels)



def train_one_epoch(model : nn.Module, optimizer , loss_fn, loader : DataLoader, device : torch.device):
    model.train()
    cumulative_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    recall = torchmetrics.Recall(task="multiclass", num_classes=10)
    precision = torchmetrics.Precision(task="multiclass", num_classes=10)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=10)
    auc = torchmetrics.AUROC(task="multiclass", num_classes=10)

    for images, labels in loader:
        # print("hello")
        images = images.to(device)
        y = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # print("aurgh")
            probs = torch.sigmoid(logits)
            cumulative_loss += len(labels) * loss.item()
            # print("a")
            accuracy.update(probs, y)
            recall.update(probs, y)
            precision.update(probs, y)
            f1.update(probs, y)
            auc.update(probs, y)

    return {
        "loss" : cumulative_loss / len(loader.dataset),
        "accuracy" : accuracy.compute().item(),
        "recall" : recall.compute().item(),
        "precision" : precision.compute().item(),
        "f1" : f1.compute().item(),
        "auc" : auc.compute().item(),

    }


def validate(model : nn.Module, loss_fn, loader : DataLoader, device : torch.device):
    model.eval()
    cumulative_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    recall = torchmetrics.Recall(task="multiclass", num_classes=10)
    precision = torchmetrics.Precision(task="multiclass", num_classes=10)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=10)
    auc = torchmetrics.AUROC(task="multiclass", num_classes=10)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            y = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, y)

            probs = torch.sigmoid(logits)
            cumulative_loss += len(labels) * loss.item()
            accuracy.update(probs, y)
            recall.update(probs, y)
            precision.update(probs, y)
            f1.update(probs, y)
            auc.update(probs, y)

    return {
        "loss" : cumulative_loss / len(loader.dataset),
        "accuracy" : accuracy.compute().item(),
        "recall" : recall.compute().item(),
        "precision" : precision.compute().item(),
        "f1" : f1.compute().item(),
        "auc" : auc.compute().item(),

    }

    


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    run = wandb.init(project="ResNet", config = {
        "num_epochs" : 100,
        "model" : {
            "architecture" : "resnet_18",
            "checkpoint_path" : "checkpoints/"
            
        },

        "optimizer" : {
            "lr" : 0.05,
            "betas" : (0.9,0.99)
        },
    })
    config = run.config
    current_epoch = 0

    model = DEEPCNN(inner_channels=4, num_inner_blocks=3).to(device)
    run.watch(models=model, log="gradients", log_freq=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"])
    loss_fn = nn.CrossEntropyLoss()

    checkpoint = config["model"]["checkpoint_path"] + "latest.pth"
    os.makedirs(config["model"]["checkpoint_path"], exist_ok=True)
    if os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location = device)
        model.load_state_dict(checkpoint_dict["model_state"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        current_epoch = checkpoint_dict["checkpoint_epoch"]


    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(32,32)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle = True)

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(32,32)),
        transforms.ToTensor(),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=32, collate_fn=collate_fn, shuffle = False)

    for epoch in range(current_epoch, config["num_epochs"]):

        if epoch % 10 == 0:
            state_dict = {
                "model_state" : model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "checkpoint_epoch" : epoch,
            }
            torch.save(f=checkpoint, obj=state_dict)

            artifact = wandb.Artifact(name= "model_gradients", type="model", metadata={"epoch" : epoch})
            artifact.add_file(checkpoint)
            run.log_artifact(artifact)

        epoch_train = train_one_epoch(model = model, optimizer=optimizer, loss_fn=loss_fn, loader=trainloader, device = device)
        epoch_val = validate(model = model, loss_fn=loss_fn, loader=testloader, device = device)
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
