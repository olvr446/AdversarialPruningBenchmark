from newResNet18 import ResNet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet18().to(device)

# score = torch.tensor([[0.1, 2.0, -1.0],
#                      [3.0, 0.5, 0.2]])
# print(score.argmax(dim=1))

# preds  = torch.tensor([1, 0, 2, 2])
# labels = torch.tensor([1, 0, 1, 2])
# print(preds == labels)
# print((preds == labels).sum())
# print((preds == labels).sum().item())

x = torch.randn(4, 3, 32, 32).to(device)
out = model(x)
print(out.requires_grad)          # predict
with torch.no_grad():
    out = model(x)
print(out.requires_grad)          # predict