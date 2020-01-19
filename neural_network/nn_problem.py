#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:19:18 2019

@author: zafar
"""
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os


afile = open('/home/zafar/deepgaze/pitch_and_yaw.txt')
a = []
 
for line in afile:
    line = line.rstrip()
    wds = line.split()
    
    a.append(float(wds[0]))
    a.append(float(wds[1]))
    a.append(float(wds[2]))

X_train = (np.reshape(a, (246,3))).astype('float64')

stripped_idx = []
curr_last = 0
for idx in range(len(X_train)):
    if idx == curr_last:
        stripped_idx.append(True)
        curr_last = idx+3
    else:
        stripped_idx.append(False)

# print(len(X_train))
# print(len(stripped_idx))
# print("-------------")
X_train = X_train[stripped_idx]


bfile = open('/home/zafar/zafar-rtech/src/reading_angle_values/coordinates.txt')
x = []
y = []
z = []  
for line in bfile:
    line = line.rstrip()
    wds = line.split()
    if wds[0] == 'x:':
        x.append(float(wds[1]))
    if wds[0] == 'y:':
        y.append(float(wds[1]))
    if wds[0] == 'z:':
        z.append(float(wds[1]))

xyz = np.column_stack((x,y,z))
Y_test = xyz[246:,:].astype('float64')
Y_train = xyz[:246,:].astype("float32")
Y_train = Y_train[stripped_idx]
print (np.shape(Y_train))
print (np.shape(X_train))

class MLP(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3,10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(),
            torch.nn.Linear(5,3)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out
inputDim = 3
outputDim = 3
learningRate = 0.0025
epochs = 1500

model = MLP(inputDim, outputDim)
model.double()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    
    inputs = Variable(torch.from_numpy(X_train)).double()
    labels = Variable(torch.from_numpy(Y_train)).double()
    
    # model.train()
    
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs).double()

    # print(outputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels).double()
    # print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# print(outputs)

torch.save(model.state_dict(), os.path.join(os.getcwd(),'mymodel.pt'))

prediction_file = open('/home/zafar/deepgaze/pitch_and_yaw_Houman.txt')
predictions = []
 
for line in prediction_file:
    line = line.rstrip()
    wds = line.split()
    
    predictions.append(float(wds[0]))
    predictions.append(float(wds[1]))
    predictions.append(float(wds[2]))

predictions = (np.reshape(predictions, (246,3))).astype('float64')
X_test = (np.asarray(predictions)).astype('float64')

with torch.no_grad(): # we don't need gradients in the testing phase
    
    test_predicted = model(Variable(torch.from_numpy(X_test))).data.numpy()


with torch.no_grad(): # we don't need gradients in the testing phase
    
    train_predicted = model(Variable(torch.from_numpy(X_train))).data.numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_train[:,0], Y_train[:,1], Y_train[:,2], 'b')
ax.scatter(train_predicted[:,0], train_predicted[:,1], train_predicted[:,2], 'rs')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

