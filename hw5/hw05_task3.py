'''
In task 3, I used the DLStudio.DetectAndLocalize.PurdueShapes5Dataset class,
I keep the enumerate part and I modify the code of reading data,
then I combine data from all noise level to a single dataset and then 
feed into the training and testing dataloader.

For the neural network, I use a network based on ResNet with basically
similar structure with HW04, and it turns out the detection rate of 
noise level is pretty high, over 95%

For task 3, there is another task requirement which use the predicted 
noise level label to design the network logic. Honestly, I don't really
understand this part. So I just try some modification to the original
LOADnet 2, which is similar practice what I do for task 4.
'''

import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
import scipy
import DLStudio
from DLStudio import *

class my(DLStudio.DetectAndLocalize.PurdueShapes5Dataset):
    def __init__(self, dl_studio, train_or_test, transform=None, noise_detect = False):
        super(DLStudio.DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
        if noise_detect == True:
            if train_or_test == 'train':
                datasetall = {}
                for dataset_file in ["PurdueShapes5-10000-train.gz",
                "PurdueShapes5-10000-train-noise-20.gz",
                "PurdueShapes5-10000-train-noise-50.gz",
                "PurdueShapes5-10000-train-noise-80.gz"
                ]:
                    if dataset_file == "PurdueShapes5-10000-train.gz" or dataset_file == "PurdueShapes5-1000-test.gz":
                        noise = 0
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_0 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-20.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-20.gz":
                        noise = 1
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_1 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-50.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-50.gz":
                        noise = 2
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_2 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-80.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-80.gz":
                        noise = 3
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_3 = dataset_new
                        
                for k in range(10000):
                    datasetall[k] = dataset_0[k]
                for k in range(10000):
                    datasetall[k+10000] = dataset_1[k]
                for k in range(10000):
                    datasetall[k+20000] = dataset_2[k]
                for k in range(10000):
                    datasetall[k+30000] = dataset_3[k]
                    
                #print(len(datasetall))
                self.dataset = datasetall
            
            if train_or_test == 'test':
                datasetall = {}
                for dataset_file in ["PurdueShapes5-1000-test.gz",
                "PurdueShapes5-1000-test-noise-20.gz",
                "PurdueShapes5-1000-test-noise-50.gz",
                "PurdueShapes5-1000-test-noise-80.gz"]:
                    if dataset_file == "PurdueShapes5-10000-train.gz" or dataset_file == "PurdueShapes5-1000-test.gz":
                        noise = 0
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_0 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-20.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-20.gz":
                        noise = 1
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_1 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-50.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-50.gz":
                        noise = 2
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_2 = dataset_new
                    if dataset_file == "PurdueShapes5-10000-train-noise-80.gz" or  \
                    dataset_file == "PurdueShapes5-1000-test-noise-80.gz":
                        noise = 3
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset_new = f.read()
                        if sys.version_info[0] == 3:
                            dataset_new,label_map = pickle.loads(dataset_new, encoding='latin1')
                        else:
                            dataset_new,label_map = pickle.loads(dataset_new)
                        for i in range(len(dataset_new)):
                            dataset_new[i].append(noise)
                        dataset_3 = dataset_new
                        
                for k in range(1000):
                    datasetall[k] = dataset_0[k]
                for k in range(1000):
                    datasetall[k+1000] = dataset_1[k]
                for k in range(1000):
                    datasetall[k+2000] = dataset_2[k]
                for k in range(1000):
                    datasetall[k+3000] = dataset_3[k]
                    
                #print(len(datasetall))
                self.dataset = datasetall
            
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = np.array( self.dataset[idx][0] )
        g = np.array( self.dataset[idx][1] )
        b = np.array( self.dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
       

        sample = {'image' : im_tensor, 
                  'bbox' : bb_tensor,
                  'label' : self.dataset[idx][4],
                  'noise_level' : self.dataset[idx][5]}
        
        return sample
        
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


class NoiseDetectNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=4):
        super(NoiseDetectNet, self).__init__()
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

def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['noise_level']
            #inputs, labels = data['image'], data['label']
            #print(i)
            #print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 10000 == 9999:
                print("\n[epoch %d: %.3f]" %
                (epoch + 1, running_loss / float(10000)))
                running_loss = 0.0
            



def run_code_for_testing(net):
    a = torch.from_numpy(np.zeros((5, 5), dtype=np.float32))
    with torch.no_grad():
        total = 0
        correct = 0
        for data in test_dataloader:
            images, labels = data['image'], data['noise_level']
            #images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            for i in range(4):
                total += 1
                a[labels[i]][predicted[i]] += 1
                if labels[i] == predicted[i]:
                    correct += 1

    print("Classification Accuracy: %.3f%%" % (correct/total*100))
    print(a)


import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


dls = DLStudio(
                  dataroot = "/home/xu1363/Downloads/DLStudio-1.1.0/Examples/data/",
                  image_size = [32,32],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 2,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


detector = DLStudio.DetectAndLocalize( dl_studio = dls )
dataserver_train = my(
                                   train_or_test = 'train',
                                   dl_studio = dls,
                                   noise_detect = True,
#                                   dataset_file = "PurdueShapes5-20-train.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-20.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-50.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz", 
                                                                      )
dataserver_test = my(
                                   train_or_test = 'test',
                                   dl_studio = dls,
                                   noise_detect = True,
#                                   dataset_file = "PurdueShapes5-20-test.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz"
                                                                  )
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

train_dataloader = torch.utils.data.DataLoader(detector.dataserver_train,
                    batch_size=4,shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(detector.dataserver_test,
                    batch_size=4,shuffle=False, num_workers=0)

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NoiseDetectNet(SkipBlocks)
run_code_for_training(model)
run_code_for_testing(model)


