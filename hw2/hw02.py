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
    
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data_loc = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data_loc = dset.CIFAR10(root='./data', train=False, download=True, transform=transform)
class data_loc_custom(dset.CIFAR10):
    def __init__(self,data_loc):
        self.samples = []
        for i in range(len(data_loc)):
            if data_loc[i][1] == 3:
                self.samples.append((data_loc[i][0],0))
            elif data_loc[i][1] == 5:
                self.samples.append((data_loc[i][0],1))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

train_data_loc_custom = data_loc_custom(train_data_loc)
test_data_loc_custom = data_loc_custom(test_data_loc)
train_loader = torch.utils.data.DataLoader(train_data_loc_custom, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data_loc_custom, batch_size=4, shuffle=False, num_workers=0)

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N, D_in, H1, H2, D_out = 8, 3*32*32, 1000, 256, 2
# Randomly initialize weights
w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
w2 = torch.randn(H1, H2, device=device, dtype=dtype)
w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
learning_rate = 1e-9

for t in range(50):
    loss_all = 0
    for i, data in enumerate(train_loader):
        #print(i)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        x = inputs.view(inputs.size(0), -1)
        h1 = x.mm(w1) ## In numpy, you would say h1 = x.dot(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        # Compute and print loss
        y = labels.view(labels.size(0), -1)
        loss = (y_pred - y).pow(2).sum().item()
        #if t % 5 == 1:
            #print(t, loss)
        loss_all += loss
        y_error = y_pred - y
        grad_w3 = h2_relu.t().mm(2 * y_error) #<<<<<< Gradient of Loss w.r.t w3
        h2_error = 2.0 * y_error.mm(w3.t()) # backpropagated error to the h2 hidden layer
        h2_error[h2 < 0] = 0 # We set those elements of the backpropagated error
        grad_w2 = h1_relu.t().mm(2 * h2_error) #<<<<<< Gradient of Loss w.r.t w2
        h1_error = 2.0 * h2_error.mm(w2.t()) # backpropagated error to the h1 hidden layer
        h1_error[h1 < 0] = 0 # We set those elements of the backpropagated error
        grad_w1 = x.t().mm(2 * h1_error) #<<<<<< Gradient of Loss w.r.t w2
        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
    print('Epoch ',t,':',' ',loss_all/(i+1), sep='')
print('')

correct = 0
total = 0

for i, data in enumerate(test_loader):
        #print(i)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        x = inputs.view(inputs.size(0), -1)
        h1 = x.mm(w1) ## In numpy, you would say h1 = x.dot(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        # Compute and print loss
        y = labels.view(labels.size(0), -1)
        #print(y)
        y_predicted = torch.max(y_pred, 1)[1]
        #print(y_predicted)
        for i in range(4):
            total += 1
            if y[i][0] == y_predicted[i]:
                correct += 1
        #time.sleep(1)
print('\nTest Accuracy : ', correct/total*100,'%',sep='')