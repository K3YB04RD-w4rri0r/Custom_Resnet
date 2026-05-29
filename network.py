import torch.nn as nn

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
        

