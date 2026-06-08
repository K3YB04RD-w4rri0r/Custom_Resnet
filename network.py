import torch.nn as nn
import torch
from typing import List

class CNNBlock(nn.Module):
    def __init__(self, in_channels : int = 1,
                 out_channels : int = 1,
                 ):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.normalization_fn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding="same", padding_mode= "zeros")
        # self.pool = nn.MaxPool2d(in_channels= out_channels, kernel_size=)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.normalization_fn(x)
        x= self.lrelu(x)
        return x
    


class MLPHead(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(in_features=input_size, out_features = 32)
        self.layer2 = nn.Linear(32 , output_size)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.lrelu(x)
        x = self.layer2(x)

        return x 



class DEEPCNN(nn.Module):
    def __init__(self, in_channels, inner_channels, num_inner_blocks: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer = CNNBlock(in_channels=in_channels, out_channels=inner_channels)
        self.blocks = nn.ModuleList([
                            CNNBlock(inner_channels, inner_channels)
                            for _ in range(num_inner_blocks)])
        self.depth = num_inner_blocks
        self.lrelu = nn.LeakyReLU()
        self.final_layer = MLPHead(input_size= inner_channels * 32 * 32, output_size=10)
        

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        return x
        

class BasicResBlock(nn.Module):
    def __init__(self, in_channels : int, red_channels  : int, stride : int = 1, expansion : int = 1):
        super().__init__()
        self.normalization_fn1 = nn.BatchNorm2d(num_features=red_channels)
        self.normalization_fn2 = nn.BatchNorm2d(num_features=red_channels * expansion)
        self.relu = nn.ReLU()

        self.in_channels = in_channels
        self.red_channels = red_channels
        self.expansion = expansion
        self.stride = stride



        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=red_channels
                                , kernel_size=3, padding = 3 // 2, padding_mode="zeros")
        
        if self.in_channels != self.red_channels * self.expansion or self.stride != 1:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=red_channels * expansion
                                    , kernel_size=1, padding = 1 // 2, padding_mode="zeros", stride = stride)
        else:
            self.shortcut = nn.Identity()

        self.layer2 = nn.Conv2d(in_channels=red_channels, out_channels=red_channels * expansion
                                , kernel_size=3, padding = 3 // 2, padding_mode="zeros", stride = stride)
        

    def forward(self, x : torch.Tensor):

        skip = self.shortcut(x)


        x = self.layer1(x)
        x = self.normalization_fn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.normalization_fn2(x) + skip
        x = self.relu(x)

        return x
    

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels : int, red_channels : int, expansion : int = 1, stride : int = 1):
        super().__init__()
        self.normalization_fn1 = nn.BatchNorm2d(num_features=red_channels)
        self.normalization_fn2 = nn.BatchNorm2d(num_features=red_channels)
        self.normalization_fn3 = nn.BatchNorm2d(num_features=red_channels * expansion)
        self.relu = nn.ReLU()

        self.in_channels = in_channels
        self.red_channels = red_channels
        self.expansion = expansion
        self.stride = stride

        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=red_channels
                                , kernel_size=1, padding = 1 // 2, padding_mode="zeros")
        
        if self.in_channels != self.red_channels * expansion or self.stride != 1:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=red_channels * expansion
                                    , kernel_size=1, padding = 1 // 2, padding_mode="zeros", stride = stride)
        else:
            self.shortcut = nn.Identity()

        self.layer2 = nn.Conv2d(in_channels=red_channels, out_channels=red_channels
                                , kernel_size=3, padding = 3 // 2, padding_mode="zeros", stride = stride)
        
        self.layer3 =  nn.Conv2d(in_channels=red_channels, out_channels=red_channels * expansion
                                , kernel_size=1, padding = 1 // 2, padding_mode="zeros")
    

    def forward(self, x : torch.Tensor):

        skip = self.shortcut(x)


        x = self.layer1(x)
        x = self.normalization_fn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.normalization_fn2(x) 
        x = self.relu(x)
        x = self.layer3(x)
        x = self.normalization_fn3(x) + skip
        x = self.relu(x)
        return x



class CustomResNet(nn.Module):
    def __init__(self, in_channels : int, num_classes : int,  schema : List[tuple[int, int, int, int]], block_type : BasicResBlock):
        super().__init__()
        self.block_type = block_type
        self.construction_schema = schema

        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels = 64, kernel_size = 7, stride=2, padding = 7//2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1))
        
        self.inner_blocks, self.inner_blocks_output_channels = self.__make__()
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=self.inner_blocks_output_channels, out_features=num_classes),
        )


    def __make__(self):
        modules = []
        input_channels = 64 # applied after the first 7*7 convolution
        for blocks, red_channels, stride, expansion in self.construction_schema:
            for i in range(blocks):
                modules.append(self.block_type(in_channels=input_channels, red_channels=red_channels, stride = stride if i == 0 else 1, expansion=expansion))
                input_channels = red_channels * expansion
        return nn.Sequential(*modules), input_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.inner_blocks(x)
        x = self.head(x)

        return x






    
def parameter_count(model : nn.Module):
    trainable_params = 0
    for _, p in model.named_parameters():
        trainable_params += p.numel() 
    return trainable_params




if __name__ == "__main__":
    model = CustomResNet(
    in_channels=3,
    num_classes=10,
    schema=[(3,64,1,4),(4,128,2,4),(6,256,2,4),(3,512,2,4)],
    block_type=BottleneckBlock)
    x = torch.randn(5, 3, 224, 224)
    out = model(x)
    assert out.shape == (5, 10), f"Expected (5,10), got {out.shape}"

    print(parameter_count(model))