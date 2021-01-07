'''
For task 4, I use the class of LOADnet2 to build my own modified logic,
so basically I make some change over the maximum channel number 
from previous 128 to 256 in my case.

From the test result, the modified LOADnet2 has the capability to 
recognize the image with noise level of 50%
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
from scipy import ndimage
import DLStudio
from DLStudio import *

class myLOADnet2_Noise50(DLStudio.DetectAndLocalize.LOADnet2):
    def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.DetectAndLocalize.LOADnet2, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.skip128to256 = DLStudio.SkipConnections.SkipBlock(128, 256, 
                                                            skip_connections=skip_connections )
                self.skip256 = DLStudio.SkipConnections.SkipBlock(256, 256, 
                                                             skip_connections=skip_connections)
                self.skip256ds = DLStudio.SkipConnections.SkipBlock(256,256,
                                            downsample=True, skip_connections=skip_connections)
                
                self.fc1 =  nn.Linear(128//2 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)


    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv(x)))          
        ## The labeling section:
        x1 = x.clone()
        for _ in range(self.depth // 4):
            x1 = self.skip64(x1)                                               
        x1 = self.skip64ds(x1)
        for _ in range(self.depth // 4):
            x1 = self.skip64(x1)                                               
        x1 = self.skip64to128(x1)
        for _ in range(self.depth // 4):
            x1 = self.skip128(x1)                                               
        x1 = self.skip128ds(x1)                                               
        for _ in range(self.depth // 4):
            x1 = self.skip128(x1)
        
		'''
		
		'''
        x1 = self.skip128to256(x1)
        for _ in range(self.depth // 4):
            x1 = self.skip256(x1)                                               
        x1 = self.skip256ds(x1)                                               
        for _ in range(self.depth // 4):
            x1 = self.skip256(x1)
        x1 = x1.view(-1, 128//2 * (32 // 2**self.pool_count)**2 )
        x1 = torch.nn.functional.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        return x1
    

def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs)
            #print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 2500 == 2499:
                print("\n[epoch %d: %.3f]" %
                (epoch + 1, running_loss / float(2500)))
                running_loss = 0.0
            



def run_code_for_testing(net):
    a = torch.from_numpy(np.zeros((5, 5), dtype=np.float32))
    with torch.no_grad():
        total = 0
        correct = 0
        for data in test_dataloader:
            images, labels = data['image'], data['label']
            #images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
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
dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'train',
                                   dl_studio = dls,
#                                   noise_detect = True,
#                                   dataset_file = "PurdueShapes5-20-train.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-20.gz", 
                                   dataset_file = "PurdueShapes5-10000-train-noise-50.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz", 
                                                                      )
dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'test',
                                   dl_studio = dls,
#                                   noise_detect = True,
#                                   dataset_file = "PurdueShapes5-20-test.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
                                   dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
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
model = myLOADnet2_Noise50(skip_connections=True, depth=32)
run_code_for_training(model)
run_code_for_testing(model)