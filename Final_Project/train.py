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

def basic_train(model, loader, device):
    for epoch in range(5):
        loss_tracker = 0
        num_batches = len(loader)
        model.train()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            loss_tracker += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, loader, device)
        print(epoch, loss_tracker/num_batches, acc)

def evaluate(model, loader, device):
    model.eval()
    correct_pred_count = 0
    total_img_count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            predictions = output.argmax(dim=1)
            total_img_count += len(inputs)
            correct_pred_count += (predictions == labels).sum().item()
    accuracy_pct = (correct_pred_count / total_img_count) * 100
    return accuracy_pct

def fgsm_attack(model, inputs, labels, eps):
    model.eval()
    inpts = inputs.detach().requires_grad_(True)
    output = model(inpts)

    loss = criterion(output, labels)
    g, = torch.autograd.grad(loss, inpts)
    inpts_step = inputs + torch.sign(g) * eps
    inpts_adv = inpts_step.clamp(0.0, 1.0)

    model.train()
    return inpts_adv.detach()

def pgd_attack(model, inputs, labels, eps, alpha, steps):
    delta = torch.tensor([-eps, eps])


basic_train(model, small_loader, device)









