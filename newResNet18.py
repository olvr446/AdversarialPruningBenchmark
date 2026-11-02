import torch
from torch import nn
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import time
# import datetime
# import os

tensor = torch.randn(8, 3, 32, 32)
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
print(conv(tensor).shape)

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if (in_channel != out_channel) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channel)
            )