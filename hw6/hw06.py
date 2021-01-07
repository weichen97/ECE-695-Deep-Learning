'''
In task 1, I tried several different alternatives other than one-hot encoding, 
like say using index of word in the sorted vocabulary to represent the word, 
furthermore, I modified index value as scaling the index value to 0-1, as depicted in 
Gabrielloye's blog which would achieve better performance. 
However, for all methods I described above, none of them can really make a difference 
than the original one-hot method, in the sense of making training loss decrease using GRU,
i.e. the network learns nothing. What I observe from my experiment is the training loss
oscillating around 0.7, while the predictions are always predicted positive, for which 
I've been puzzled with, I haven't figured out why this doesn't really work.

In task 2, I tried combine the input with the (t-1) hidden state as well as (t-1) input state,
also using different nonlinearity to replace tanh. For word representation I tried both one-hot enocidng
and word index. For test results, the modified network still shows no apparent progress in learning, i.e.
the loss doesn't decrease, the prediction basically doesn't improve much. 

In task 3, I pad all input review length to a constant number, then tried training the network
in batch. The training time for iteration apparently increase due to the added length of input.
Still, as the previous tasks, training in batch seems unable to improve the network in my case using different
gated networks. 
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

'''
In HW06, most of my code implementation is based on DLStudio.TextClassification class of DLStudio 1.1.3,
it saves much efforts to use the provided training function instead of starting from scratch.
The changes I made to the class includes changing the word representation to word index representation,
changing the structure of the TEXTnetOrder2, padding the input review length to a constant number to 
enable the batch processing.
'''
class my(DLStudio.TextClassification.SentimentAnalysisDataset):
    def __init__(self, dl_studio, train_or_test, dataset_file):
        super(DLStudio.TextClassification.SentimentAnalysisDataset, self).__init__()
        self.train_or_test = train_or_test
        root_dir = dl_studio.dataroot
        f = gzip.open(root_dir + dataset_file, 'rb')
        dataset = f.read()
        if train_or_test is 'train':
            if sys.version_info[0] == 3:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
            self.categories = sorted(list(self.positive_reviews_train.keys()))
            self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
            self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
            self.indexed_dataset_train = []
            for category in self.positive_reviews_train:
                for review in self.positive_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 1])
            for category in self.negative_reviews_train:
                for review in self.negative_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 0])
            random.shuffle(self.indexed_dataset_train)
        elif train_or_test is 'test':
            if sys.version_info[0] == 3:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
            self.vocab = sorted(self.vocab)
            self.categories = sorted(list(self.positive_reviews_test.keys()))
            self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
            self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
            self.indexed_dataset_test = []
            for category in self.positive_reviews_test:
                for review in self.positive_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 1])
            for category in self.negative_reviews_test:
                for review in self.negative_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 0])
            random.shuffle(self.indexed_dataset_test)

    def get_vocab_size(self):
        return len(self.vocab)
    
    '''
    Here I change the original one-hot vector representation of words as the 
    index of word in the sorted vocabulary.
    '''
    def one_hotvec_for_word(self, word):
        word_index =  self.vocab.index(word)/len(self.vocab)
        #word_index =  self.vocab.index(word)
        return word_index

    def review_to_tensor(self, review):
        #review_tensor = torch.zeros(len(review), len(self.vocab))
        review_tensor = torch.zeros(len(review), 1)
        for i,word in enumerate(review):
            review_tensor[i,:] = self.one_hotvec_for_word(word)
        return review_tensor

    def sentiment_to_tensor(self, sentiment):
        '''
        Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
        sentiment and 1 for positive sentiment.  We need to pack this value in a
        two-element tensor.
        '''      
        sentiment_tensor = torch.zeros(2)
        if sentiment is 1:
            sentiment_tensor[1] = 1
        elif sentiment is 0: 
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        if self.train_or_test is 'train':
            return len(self.indexed_dataset_train)
        elif self.train_or_test is 'test':
            return len(self.indexed_dataset_test)

    def __getitem__(self, idx):
        sample = self.indexed_dataset_train[idx] if self.train_or_test is 'train' else self.indexed_dataset_test[idx]
        review = sample[0]
        review_category = sample[1]
        review_sentiment = sample[2]
        review_sentiment = self.sentiment_to_tensor(review_sentiment)
        review_tensor = self.review_to_tensor(review)
        category_index = self.categories.index(review_category)
        sample = {'review'       : review_tensor, 
                  'category'     : category_index, # should be converted to tensor, but not yet used
                  'sentiment'    : review_sentiment }
        return sample
    

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
#                  dataroot = "/home/kak/TextDatasets/",
                  dataroot = "./data/",
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
#                  learning_rate =  0.004,
                  learning_rate =  1e-4,
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = my(
                                 train_or_test = 'train',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
                                dataset_file = "sentiment_dataset_train_200.tar.gz", 
#                                 dataset_file = "sentiment_dataset_train_40.tar.gz", 
                                                                      )
dataserver_test = my(
                                 train_or_test = 'test',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
                                dataset_file = "sentiment_dataset_test_200.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

'''
Changing the feature number from vocabulary length to 1, since only word index matters
'''
#vocab_size = dataserver_train.get_vocab_size()
vocab_size = 1
hidden_size = 512
output_size = 2                            # for positive and negative sentiments
n_layers = 2

model = text_cl.GRUnet(vocab_size, hidden_size, output_size, n_layers)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

## TRAINING:
print("\nStarting training --- BE VERY PATIENT, PLEASE!  The first report will be at 100th iteration. May take around 5 minutes.\n")
text_cl.run_code_for_training_for_text_classification_with_gru(model, hidden_size)


text_cl.run_code_for_testing_text_classification_with_gru(model, hidden_size)


'''
I changed the structure of TEXTnetOrder2, combing the input with (t-1) state input
'''
class mytestnet(DLStudio.TextClassification.TEXTnetOrder2):
    def __init__(self, input_size, hidden_size, output_size, dls):
        super(DLStudio.TextClassification.TEXTnetOrder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
        self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
        self.middle_to_out = nn.Linear(100, output_size)     
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        # for the cell
        self.cell = torch.zeros(1, hidden_size).to(dls.device)
        self.linear_for_cell = nn.Linear(hidden_size, hidden_size)
        self.cell2 = torch.zeros(1, input_size).to(dls.device)
        self.linear_for_cell2 = nn.Linear(input_size, input_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden, self.cell), 1)
        combined = torch.cat((input, hidden, self.cell, self.cell2), 1)
        hidden = self.combined_to_hidden(combined)
        out = self.combined_to_middle(combined)
        out = torch.nn.functional.relu(out)
        out = self.dropout(out)
        out = self.middle_to_out(out)
        out = self.logsoftmax(out)
        hidden_clone = hidden.clone()
        input_clone = input.clone()
        #self.cell = torch.tanh(self.linear_for_cell(hidden_clone))
        self.cell = self.sigmoid(self.linear_for_cell(hidden_clone))
        self.cell2 = self.sigmoid(self.linear_for_cell2(input_clone))
        return out,hidden

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
#                  dataroot = "/home/kak/TextDatasets/",
                  dataroot = "./data/",
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  0.004,  
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = my(
                                 train_or_test = 'train',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
                                dataset_file = "sentiment_dataset_train_40.tar.gz",
                                                                      )
dataserver_test = my(
                                 train_or_test = 'test',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()
vocab_size = 1
hidden_size = 512
output_size = 2                            # for positive and negative sentiments

model = mytestnet(vocab_size, hidden_size, output_size,dls)

##  DO NOT UNCOMMENT THE NEXT LINE UNLESS YOU KNOW WHAT YOU ARE DOING.
#model = text_cl.TEXTnetOrder2(vocab_size, hidden_size, output_size, dls)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

text_cl.run_code_for_training_for_text_classification_no_gru(model, hidden_size)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    text_cl.run_code_for_testing_text_classification_no_gru(model, hidden_size)


class my2(DLStudio.TextClassification.SentimentAnalysisDataset):
    def __init__(self, dl_studio, train_or_test, dataset_file):
        super(DLStudio.TextClassification.SentimentAnalysisDataset, self).__init__()
        self.train_or_test = train_or_test
        root_dir = dl_studio.dataroot
        f = gzip.open(root_dir + dataset_file, 'rb')
        dataset = f.read()
        if train_or_test is 'train':
            if sys.version_info[0] == 3:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
            self.categories = sorted(list(self.positive_reviews_train.keys()))
            self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
            self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
            self.indexed_dataset_train = []
            for category in self.positive_reviews_train:
                for review in self.positive_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 1])
            for category in self.negative_reviews_train:
                for review in self.negative_reviews_train[category]:
                    self.indexed_dataset_train.append([review, category, 0])
            random.shuffle(self.indexed_dataset_train)
        elif train_or_test is 'test':
            if sys.version_info[0] == 3:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
            else:
                self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
            self.vocab = sorted(self.vocab)
            self.categories = sorted(list(self.positive_reviews_test.keys()))
            self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
            self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
            self.indexed_dataset_test = []
            for category in self.positive_reviews_test:
                for review in self.positive_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 1])
            for category in self.negative_reviews_test:
                for review in self.negative_reviews_test[category]:
                    self.indexed_dataset_test.append([review, category, 0])
            random.shuffle(self.indexed_dataset_test)

    def get_vocab_size(self):
        return len(self.vocab)
    
    '''
    Here I change the original one-hot vector representation of words as the 
    index of word in the sorted vocabulary.
    '''
    def one_hotvec_for_word(self, word):
        word_index =  self.vocab.index(word)/len(self.vocab)
        #word_index =  self.vocab.index(word)
        return word_index

    def review_to_tensor(self, review):
        #review_tensor = torch.zeros(len(review), len(self.vocab))
        '''
		I make each review to have the same length 2500
		'''
		review_tensor = torch.zeros(2500, 1)
        for i,word in enumerate(review):
            review_tensor[i,:] = self.one_hotvec_for_word(word)
        #for j in range(2000-i):
        #    review_tensor[i+j+1,:] = 0
        return review_tensor

    def sentiment_to_tensor(self, sentiment):
        """
        Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
        sentiment and 1 for positive sentiment.  We need to pack this value in a
        two-element tensor.
        """        
        sentiment_tensor = torch.zeros(2)
        if sentiment is 1:
            sentiment_tensor[1] = 1
        elif sentiment is 0: 
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        if self.train_or_test is 'train':
            return len(self.indexed_dataset_train)
        elif self.train_or_test is 'test':
            return len(self.indexed_dataset_test)

    def __getitem__(self, idx):
        sample = self.indexed_dataset_train[idx] if self.train_or_test is 'train' else self.indexed_dataset_test[idx]
        review = sample[0]
        review_category = sample[1]
        review_sentiment = sample[2]
        review_sentiment = self.sentiment_to_tensor(review_sentiment)
        review_tensor = self.review_to_tensor(review)
        len = review_tensor.shape[0]
        #print(review_tensor.shape)
        category_index = self.categories.index(review_category)
        #print(review_sentiment.shape)
        sample = {'review'       : review_tensor, 
                  'category'     : category_index, # should be converted to tensor, but not yet used
                  'sentiment'    : review_sentiment }
        return sample



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
#                  dataroot = "/home/kak/TextDatasets/",
                  dataroot = "./data/",
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
#                  learning_rate =  0.004,
                  learning_rate =  1e-4,
                  epochs = 1,
                  batch_size = 2,
                  classes = ('negative','positive'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = my2(
                                 train_or_test = 'train',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
#                                dataset_file = "sentiment_dataset_train_200.tar.gz", 
                                 dataset_file = "sentiment_dataset_train_40.tar.gz", 
                                                                      )
dataserver_test = my2(
                                 train_or_test = 'test',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
#                                dataset_file = "sentiment_dataset_test_200.tar.gz",
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

#vocab_size = dataserver_train.get_vocab_size()
vocab_size = 1
hidden_size = 512
output_size = 2                            # for positive and negative sentiments
n_layers = 2

model = text_cl.GRUnet(vocab_size, hidden_size, output_size, n_layers)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

## TRAINING:
print("\nStarting training --- BE VERY PATIENT, PLEASE!  The first report will be at 100th iteration. May take around 5 minutes.\n")
text_cl.run_code_for_training_for_text_classification_with_gru(model, hidden_size)

## TESTING:
import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    text_cl.run_code_for_testing_text_classification_with_gru(model, hidden_size)