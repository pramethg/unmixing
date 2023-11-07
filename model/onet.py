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
            padding_mode = 'zeros')
    
    def forward(self, x):
        x = F.elu((self.conv3x3(self.in_channels, self.out_channels)(x)), alpha = 1.0)
        x = F.elu((self.conv3x3(self.out_channels, self.out_channels)(x)), alpha = 1.0)
        return x

def convtest():
    block = ConvBlock(10, 16).float().cpu()
    x = torch.rand(size = (1, 10, 396, 100)).float().cpu()
    print(block.forward(x).detach().numpy().shape)

class ONet(nn.Module):
    def __init__(self, nwaves = 10, features = [2**n for n in range(4, 9)]):
        super(ONet, self).__init__()
        self.nwaves = nwaves
        self.features = features

    @staticmethod
    def conv1x1(in_channels, out_channels = 1):
        return nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = (1, 1),
            stride = 1,
            padding = 0)

    @staticmethod
    def convt2d(in_channels, out_channels):
        return nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (2, 2),
                stride = 2)
    
    @staticmethod
    def downsample(x, features = [2**n for n in range(4, 9)]):
        skiplist = []
        x = F.elu(ConvBlock.conv3x3(features[0], features[0]).forward(x), alpha = 1.0)
        skiplist.append(x)
        for (cin, cout) in zip(features, features[1:]):
            print(f"{cin}, {cout}")
            x = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)(x)
            x = ConvBlock(cin, cout).forward(x)
            if cout < features[-1]:
                skiplist.append(x)
        return x, skiplist
    
    @staticmethod
    def upsample(x, features = [2**n for n in range(8, 3, -1)], skiplist = []):
        for idx, (cin, cout) in enumerate(zip(features, features[1:])):
            x = ONet.convt2d(cin, cout)(x)
            x = torch.cat([x, skiplist[idx]], )
            x = ConvBlock()

    def forward(self, x):
        seglist, so2list = [], []
        seg, so2 = ConvBlock(self.nwaves, self.features[0]).forward(x), ConvBlock(self.nwaves, self.features[0]).forward(x)
        seg, seglist = self.downsample(seg)
        seg = self.upsample(seg, skiplist = seglist[::-1])
        so2, so2list = self.downsample(so2)
        so2 = self.upsample(so2, skiplist = so2list[::-1])
        return seg, so2
        
    def evalforward(self, x):
        return x

def test():
    model = ONet()
    x = torch.rand(size = (1, model.nwaves, 128, 128))
    s1, s2, s3, s4 = model.forward(x)
    print(s1.shape, len(s3))

if __name__ == "__main__":
    test()