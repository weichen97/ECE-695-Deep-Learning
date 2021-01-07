import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as tvt
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets

transform = tvt.Compose(
    [tvt.ToTensor(),
     tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_data_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                (epoch + 1, i + 1, running_loss / float(2000)))
                running_loss = 0.0

import torch.nn as nn
import torch.nn.functional as F
class TemplateNet(nn.Module):
    def __init__(self):
        super(TemplateNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3,padding=(1, 1)) ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*7*7, 1000) ## (C)
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        x = self.pool(F.relu(self.conv2(x))) ## (D)
        #print(x.size())
        x = x.view(-1, 128*7*7) ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
Net = TemplateNet()
run_code_for_training(Net)


a = torch.from_numpy(np.zeros((10, 10), dtype=np.float32))
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        outputs = Net(images)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        for i in range(4):
            a[labels[i]][predicted[i]] += 1
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
print(a)