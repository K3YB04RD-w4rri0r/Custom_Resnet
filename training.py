import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from network import DEEPCNN, CustomResNet, BottleneckBlock, BasicResBlock
from typing import List
import torch.nn as nn
import wandb
import os
import torchmetrics
from typing import Sized

def collate_fn(batch_list : List[tuple[torch.Tensor, int]]):
    images, labels = zip(*batch_list)
    return torch.stack(images), torch.tensor(labels)



def train_one_epoch(model : nn.Module, optimizer , loss_fn, loader : DataLoader, device : str):
    model.train()
    cumulative_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=10).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=10).to(device)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=10).to(device)
    auc = torchmetrics.AUROC(task="multiclass", num_classes=10).to(device)

    for images, labels in loader:
        # print("hello")
        images = images.to(device)
        y = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, y)
        loss.backward()
        """
        for name, p in model.named_parameters():
            if p.grad is not None:
                print(f"{name:45s} {p.grad.abs().mean().item():.3e}")
        import sys
        sys.exit()  
        """
        optimizer.step()

        with torch.no_grad():
            # print("aurgh")
            probs = torch.softmax(logits, dim=1)
            cumulative_loss += len(labels) * loss.item()
            # print("a")
            accuracy.update(probs, y)
            recall.update(probs, y)
            precision.update(probs, y)
            f1.update(probs, y)
            auc.update(probs, y)
    assert isinstance(loader.dataset, Sized)
    return {
        "loss" : cumulative_loss / len(loader.dataset),
        "accuracy" : accuracy.compute().item(),
        "recall" : recall.compute().item(),
        "precision" : precision.compute().item(),
        "f1" : f1.compute().item(),
        "auc" : auc.compute().item(),

    }


def validate(model : nn.Module, loss_fn, loader : DataLoader, device : str):
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
        "loss" : cumulative_loss / len(loader.dataset),
        "accuracy" : accuracy.compute().item(),
        "recall" : recall.compute().item(),
        "precision" : precision.compute().item(),
        "f1" : f1.compute().item(),
        "auc" : auc.compute().item(),

    }

    


def main():
    run = wandb.init(project="ResNet", config = {
        "num_epochs" : 50,
        "seed" : 0,
        "gpu_id" : 1,
        "model" : {
            "checkpoint_path" : "checkpoints/",
            "architecture" : "Resnet_50",
            "in_channels" : 3,
            "num_classes" : 10,
            "schema" : [(3,64,1,4),(4,128,2,4),(6,256,2,4),(3,512,2,4)],
            "block_type" : "bottleneck"
            
        },

        "optimizer" : {
            "lr" :  1e-3,
            "betas" : (0.9,0.999)
        },
    })

    
    config = run.config
    device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["seed"])
    current_epoch = 0
    
    resnet_model = CustomResNet(in_channels=config["model"]["in_channels"], num_classes=config["model"]["num_classes"], schema = config["model"]["schema"], 
    block_type= BasicResBlock if config["model"]["block_type"] == "basic" else BottleneckBlock)
    run.watch(resnet_model, log="gradients", log_freq=100)
    resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"])
    


    """
    CNN
    """
    cnn_model = DEEPCNN(inner_channels=64, num_inner_blocks=48).to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"])
    


    loss_fn = nn.CrossEntropyLoss()

    checkpoint = config["model"]["checkpoint_path"] + f"{config['model']['architecture']}.pth"
    os.makedirs(config["model"]["checkpoint_path"], exist_ok=True)
    if os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location = device)
        resnet_model.load_state_dict(checkpoint_dict["model_state"])
        resnet_optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        current_epoch = checkpoint_dict["checkpoint_epoch"]


    transform_train = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle = True)

    transform_test = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor(),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=32, collate_fn=collate_fn, shuffle = False)

    for epoch in range(current_epoch, config["num_epochs"]):

        if epoch % 10 == 0:
            state_dict = {
                "model_state" : resnet_model.state_dict(),
                "optimizer_state" : resnet_optimizer.state_dict(),
                "checkpoint_epoch" : epoch,
            }
            torch.save(f=checkpoint, obj=state_dict)

            artifact = wandb.Artifact(name= f'{config["model"]["architecture"]}_state', type="model", metadata={"epoch" : epoch})
            artifact.add_file(checkpoint)
            run.log_artifact(artifact)
            print("Just uploaded model checkpoint")


        print(f"Current epoch : {epoch}")
        epoch_train = train_one_epoch(model = resnet_model, optimizer=resnet_optimizer, loss_fn=loss_fn, loader=trainloader, device = device)
        epoch_val = validate(model = resnet_model, loss_fn=loss_fn, loader=testloader, device = device)
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
        
        """
        And for CNN
        """
        epoch_train = train_one_epoch(model = cnn_model, optimizer=cnn_optimizer, loss_fn=loss_fn, loader=trainloader, device = device)
        epoch_val = validate(model = cnn_model, loss_fn=loss_fn, loader=testloader, device = device)
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
