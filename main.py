#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:45:52 2018

@author: nouman
"""

# PCA
# Importing the libraries
import numpy as np
import pandas as pd
import xlsxwriter
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import gc

profile = pd.DataFrame(columns=['ImageId', 'Label'])

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:]
y_train = dataset.iloc[:, 0]
X_test = pd.read_csv('test.csv')
X_train = torch.tensor(X_train.values)
X_test = torch.tensor(X_test.values)
y_train = torch.tensor(y_train.values)
print(X_train.shape)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=11, stride=11))
        self.fc = nn.Linear(512, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out

model = torch.load('digit.pt')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
X_test = torch.autograd.Variable(X_test).float()
y_train = torch.autograd.Variable(y_train).long()
X_train = torch.autograd.Variable(X_train).float()
print(len(X_train))
print(len(y_train))
batch = 42000//100
for i in range(5):
    count = 0
    for j in range(batch):
        X = X_train[count: count+100]
        y = y_train[count: count + 100]
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gc.collect()
        del outputs, loss, X, y
        count += 100

torch.save(model, 'digit.pt')    
batcht = 28000//100
outputs = [0] * 28000
print(len(outputs))
count = 0
for i in range(batcht):
    x = X_test[count:count+100]
    output = model(x)
    outputs[count:count+100] = output
    del x, output
    gc.collect()
outputs = model(X_test)
print(outputs)
_, predicted = torch.max(outputs, 1)
predicted = predicted.data.tolist()
print(len(predicted))
print(predicted[0])
counter = 1
for i in range(28000):
    ser = pd.Series([counter, predicted[counter - 1]], index = ['ImageId', 'Label'])
    profile = profile.append(ser, ignore_index = True)
    counter += 1
    
    
    
print(profile)
filename = 'mnist.xlsx'
writer = pd.ExcelWriter(filename, engine='xlsxwriter')
profile.to_excel(writer, index=False)
writer.save()