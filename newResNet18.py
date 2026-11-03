import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import time
# import datetime
# import os

class Block(nn.Module):
    def __init__(self, inChannel, outChannel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inChannel, 
            outChannel, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(outChannel)

        self.conv2 = nn.Conv2d(
            outChannel, 
            outChannel, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(outChannel)

        self.shortcut = nn.Sequential()
        if (inChannel != outChannel) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inChannel,
                    outChannel,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(outChannel)
            )
    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = output + self.shortcut(x)
        output = F.relu(output)
        return output
    
ds = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                  transform=T.ToTensor())
data = torch.tensor(ds.data).float() / 255
print(data.mean(dim=(0, 2, 3)))
print(data.std(dim=(0, 2, 3)))

class ResNet18(nn.Module):
    def __init__(self, cifar_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("stdev", torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))
        
        self.stemConv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.stemBn = nn.BatchNorm2d(64)

        self.stage1 = self.makeStage(64, 64, 1)
        self.stage2 = self.makeStage(64, 128, 2)
        self.stage3 = self.makeStage(128, 256, 2)
        self.stage4 = self.makeStage(256, 512, 2)

        self.final = nn.Linear(512, cifar_classes)

    def makeStage(self, inChannel, outChannel, stride):
        return nn.Sequential(
            Block(inChannel, outChannel, stride),
            Block(outChannel, outChannel, stride=1)
    )

    def forward(self, x):
        output = F.relu(self.stemBn(self.stemConv(x)))
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = F.adaptive_avg_pool2d(output, 1)
        output = output.flatten(1)
        output = self.final(output)
        return output

model = ResNet18()
print(model(torch.randn(8, 3, 32, 32)).shape)
print(sum(p.numel() for p in model.parameters()))
# print(model.state_dict().keys())






