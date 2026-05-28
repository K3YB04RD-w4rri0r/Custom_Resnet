import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import List, Callable
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import os
import torchmetrics

def collate_fn(batch_list : List[tuple[torch.Tensor, int]]):
    images, labels = zip(*batch_list)
    return torch.stack(images), torch.tensor(labels)


class CNNBlock(nn.Module):
    def __init__(self, in_channels : int = 1,
                 out_channels : int = 1,
                 ):
        super().__init__()
        self.relu = nn.functional.relu
        self.normalization_fn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding="same", padding_mode= "zeros")
        # self.pool = nn.MaxPool2d(in_channels= out_channels, kernel_size=)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.normalization_fn(x)
        x= self.relu(x)
        return x
    


class MLPHead(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(in_features=input_size, out_features = 32)
        self.layer2 = nn.Linear(32 , output_size)
        self.relu = nn.functional.relu
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x 



class DEEPCNN(nn.Module):
    def __init__(self, inner_channels, num_inner_blocks: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer = CNNBlock(in_channels=1, out_channels=inner_channels)
        self.blocks = nn.ModuleList([
                            CNNBlock(inner_channels, inner_channels)
                            for _ in range(num_inner_blocks)])
        self.depth = num_inner_blocks
        self.relu = nn.functional.relu
        self.final_layer = MLPHead(input_size= inner_channels * 32 * 32, output_size=10)
        

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        return x
        


def train_one_epoch(model : nn.Module, optimizer , loss_fn, loader : DataLoader, device : torch.device):
    model.train()
    for images, labels in loader:
        images = images.to(device)
        y = labels.to(device)
        
        optimizer.zero_grad()
        yhat = model(images)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()




def validate():
    pass


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
    current_epoch = 1

    model = DEEPCNN(inner_channels=4, num_inner_blocks=3).to(device)
    run.watch(models=model, log="gradients", log_freq=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"])
    loss_fn = nn.functional.cross_entropy

    checkpoint = config["model"]["checkpoint_path"] + "latest.pth"
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

        train_one_epoch(model = model, optimizer=optimizer, loss_fn=loss_fn, loader=trainloader, device = device)

        run.log(
            {
                "epoch" : epoch,
                "train_loss" : 1,
                "val_loss" : 1
            }
        )











if __name__ == "__main__":
    # test()
    main()
