from newResNet18 import ResNet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T

train_list = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
test_list = [T.ToTensor()]
train_compose = T.Compose(train_list)
test_compose = T.Compose(test_list)
train_set = torchvision.datasets.CIFAR10("./my_datadir/CIFAR10", train=True, transform=train_compose, download=False)
test_set = torchvision.datasets.CIFAR10("./my_datadir/CIFAR10", train=False, transform=test_compose, download=False)
small_set = Subset(train_set, range(2000))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
small_loader = DataLoader(small_set, batch_size=128, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

inputs,labels = next(iter(small_loader))
inputs = inputs.to(device)
labels = labels.to(device)
# print(inputs, labels)

output = model(inputs)
# print(output.shape)

loss = criterion(output, labels)
# print(loss)

for epoch in range(5):
    model.train()
    for inputs, labels in small_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model.forward(inputs)
        loss = criterion(output, labels)
        


