'''
In the task 2, since we are required to use the smoothing 
to the input image to improve the recognition rate,
so in the dataloader class, I add the code using Gaussian filter
function in scipy package to smooth the input image.

To experiment the optimal parameter for smoothing noised image,
I try the variance of Gaussian filter from 0.5 to 2. 
Then finally I decide that when variance equals 1.5.
To make results in all different noise level dataset consistent,
I set the variance to be 1.5 for all datasets.
In this case, all datasets have fairly acceptable recognition rate.
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


'''
I used the class of PurdueShapes5Dataset from DLStudio,
since the using the structure of original dataloader 
can save much time building from scratch.

And I made the change to the __getitem__ function,
in which I add code of smoothing using Gaussian filter 
to all R,G,B channel inputs array
'''
class my(DLStudio.DetectAndLocalize.PurdueShapes5Dataset):
    def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
        super(DLStudio.DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
        if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        else:
            root_dir = dl_studio.dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if sys.version_info[0] == 3:
                self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
            else:
                self.dataset, self.label_map = pickle.loads(dataset)
            # reverse the key-value pairs in the label dictionary:
            self.class_labels = dict(map(reversed, self.label_map.items()))
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = np.array( self.dataset[idx][0] )
        g = np.array( self.dataset[idx][1] )
        b = np.array( self.dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        
        # I add the Gaussian filter to the three channel of image
        R = scipy.ndimage.gaussian_filter(R,sigma = 1.5)
        G = scipy.ndimage.gaussian_filter(G,sigma = 1.5)
        B = scipy.ndimage.gaussian_filter(B,sigma = 1.5)
        
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        sample = {'image' : im_tensor, 
                  'bbox' : bb_tensor,
                  'label' : self.dataset[idx][4] }
        if self.transform:
             sample = self.transform(sample)
        return sample
def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['label']
            #print(i)
            #print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs[0], labels)
            #print(loss)
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
            _, predicted = torch.max(outputs[0], 1)
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


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

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
#                                   dataset_file = "PurdueShapes5-10000-train.gz"
#                                   dataset_file = "PurdueShapes5-20-train.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-20.gz", 
#                                   dataset_file = "PurdueShapes5-10000-train-noise-50.gz", 
                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz", 
                                                                      )
dataserver_test = my(
                                   train_or_test = 'test',
                                   dl_studio = dls,
#                                   dataset_file =  "PurdueShapes5-1000-test.gz"
#                                   dataset_file = "PurdueShapes5-20-test.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
#                                   dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz"
                                                                  )
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

train_dataloader = torch.utils.data.DataLoader(detector.dataserver_train,
                               batch_size=4,shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(detector.dataserver_test,
                               batch_size=4,shuffle=False, num_workers=4)
#detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

model = detector.LOADnet2(skip_connections=True, depth=32)
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run_code_for_training(model)
run_code_for_testing(model)

