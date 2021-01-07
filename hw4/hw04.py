import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tvt

class SkipBlocks(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(SkipBlocks, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)  
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:  
                layers.append(block(self.inchannel, channels, stride))
            else:    
                layers.append(block(channels, channels, 1))
            self.inchannel = channels
        return nn.Sequential(*layers)  

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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

def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(5):
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
            if i % 12000 == 11999:
                print("\n[epoch %d: %.3f" %
                (epoch + 1, running_loss / float(12000)))
                running_loss = 0.0


def run_code_for_testing(net):
    with torch.no_grad():
        total = 0
        correct = 0
        for data in test_data_loader:
            images, labels = data
			images = images.to(device)
            outputs = Net(images)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            for i in range(4):
                total += 1
                if labels[i] == predicted[i]:
                    correct += 1

    print("Classification Accuracy: %.3f%%" % (correct/total*100))
	
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Net = ResNet(SkipBlocks)
run_code_for_training(Net)
run_code_for_testing(Net)
