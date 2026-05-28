import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import List, Callable
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import os

def collate_fn(batch_list : List[tuple[torch.Tensor, int]]):
    images, labels = zip(*batch_list)
    return torch.stack(images), torch.tensor(labels)


class RESNETBlock(nn.Module):
    def __init__(self, in_channels : int = 1,
                 out_channels : int = 1,
                 kernel_conv_size : int = 3,
                 stride_conv_size : int = 1,
                 padding_conv_size : int = 0,
                 ):
        super().__init__()
        self.relu = nn.functional.relu
        self.normalization_fn = nn.functional.batch_norm()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, padding=padding_conv_size, padding_mode= "zeros")
        # self.pool = nn.MaxPool2d(in_channels= out_channels, kernel_size=)

    def forward(self, x):
        x = self.normalization_fn(x)
        x = self.conv(x)


        



class RESNET_Custom(nn.Module):
    def __init__(self, num_blocks: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer = RESNETBlock()
        self.block = RESNETBlock()
        self.depth = num_blocks
        

    def forward(self, x):
        for _ in range(self.depth):
            x = self.block(x)
        


def train_one_epoch(model : nn.Module, loader : DataLoader, device : torch.device):
    model.train()
    for  images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

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
    current_epoch = 0

    model = RESNET_Custom().to(device)
    run.watch(models=model, log="gradients", log_freq=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], betas=config["optimizer"]["betas"])

    checkpoint = config["model"]["checkpoint_path"] + "latest.pth"
    if os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location = device)
        model.load_state_dict(checkpoint_dict["model_state"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        current_epoch = checkpoint_dict["checkpoint_epoch"]


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
            run.log_artifact({
                "training_state" : artifact 
            })

        
        preds_train, loss_train = train_one_epoch()
        preds_val, loss_val = validate()

        run.log(
            {
                "epoch" : epoch,
                "train_loss" : loss_train,
                "val_loss" : loss_val
            }
        )



    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(32,32)),
        transforms.RandomRotation(20),
        transforms.PILToTensor(),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=None, shuffle = True)











if __name__ == "__main__":
    # test()
    main()
