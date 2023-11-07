import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @staticmethod
    def conv3x3(in_channels, out_channels):
        return nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = (3, 3),
            stride = 1,
            padding = 1,
            padding_mode = 'zeros',
        )
    
    def forward(self, x):
        x = F.elu((self.conv3x3(self.in_channels, self.out_channels)(x)), alpha = 1.0)
        x = F.elu((self.conv3x3(self.out_channels, self.out_channels)(x)), alpha = 1.0)
        return x

def convtest():
    block = ConvBlock(10, 16).float().cpu()
    x = torch.rand(size = (1, 10, 396, 100)).float().cpu()
    print(block.forward(x).detach().numpy().shape)

class ONet(nn.Module):
    @staticmethod
    def conv1x1():
        pass

    def __init__(self, nwaves = 10):
        super(ONet, self).__init__()
        self.nwaves = nwaves
    
    def forward(self, x):
        return x
    
    def evalforward(self, x):
        return x

def test():
    model = ONet()
    x = torch.rand(size = (1, model.nwaves, 396, 100))


if __name__ == "__main__":
    convtest()