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



def get_train_data():
    afile = open('/home/zafar/deepgaze/pitch_and_yaw.txt')
    a = []
    
    for line in afile:
        line = line.rstrip()
        wds = line.split()
        
        a.append(float(wds[0]))
        a.append(float(wds[1]))
        a.append(float(wds[2]))

    X_train = (np.reshape(a, (246,3))).astype('float64')


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
    Y_train = xyz[:246,:].astype("float32")

    afile.close()
    bfile.close()

    return X_train, Y_train


def get_test_data():
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
    Y_test = xyz[246:,:].astype("float32")

    bfile.close()
    prediction_file.close()

    return X_test, Y_test


def strip_data(X, Y):
    
    stripped_idx = []
    curr_last = 0
    for idx in range(len(X)):
        if idx == curr_last:
            stripped_idx.append(True)
            curr_last = idx+3
        else:
            stripped_idx.append(False)

    assert (len(X) == len(Y))

    return X[stripped_idx], Y[stripped_idx]


def predict_from_model(X):

    with torch.no_grad(): # we don't need gradients in the testing phase
        
        predicted = model(Variable(torch.from_numpy(X))).data.numpy()

        return predicted


class MLP(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputDim,10),
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
            torch.nn.Linear(5,outputDim)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out

inputDim = 3
outputDim = 3

model = MLP(inputDim, outputDim)
model.load_state_dict(torch.load(os.path.join(os.getcwd(),'mymodel.pt')))
model.double()

X_train, Y_train = get_train_data()

X_test, Y_test = get_test_data()

X_train, Y_train = strip_data(X_train, Y_train)

print("hereeee")
print(len(X_test))
print(len(Y_test))
X_test, Y_test = strip_data(X_test, Y_test)


# Test on train_x
predicted = predict_from_model(X_train)

fig = plt.figure()
fig.suptitle("Train Predictions")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_train[:,0], Y_train[:,1], Y_train[:,2], 'b')
ax.scatter(predicted[:,0], predicted[:,1], predicted[:,2], 'rs')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

error_train = []
for p in range(len(Y_train)):
    errors_train = math.sqrt((Y_train[p,0]-predicted[p,0])**2 + (Y_train[p,1]-predicted[p,1])**2 + (Y_train[p,2]-predicted[p,2])**2) 
    error_train.append(float(errors_train))

print(error_train)
print(len(error_train))

print ('Average error = ', np.sum(error_train)/len(error_train))
print ('Maximum error = ', np.amax(error_train))

_ = plt.hist(error_train)
plt.title('Error Train')
plt.show()


# Test on test_x
predicted = predict_from_model(X_test)

fig = plt.figure()
fig.suptitle("Test Predictions")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_test[:,0], Y_test[:,1], Y_test[:,2], 'b')
ax.scatter(predicted[:,0], predicted[:,1], predicted[:,2], 'rs')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

error_test = []
for p in range(len(Y_test)):
    errors_test = math.sqrt((Y_test[p,0]-predicted[p,0])**2 + (Y_test[p,1]-predicted[p,1])**2 + (Y_test[p,2]-predicted[p,2])**2) 
    error_test.append(float(errors_test))

print(error_test)
print(len(error_test))

print ('Average error = ', np.sum(error_test)/len(error_test))
print ('Maximum error = ', np.amax(error_test))

_ = plt.hist(error_test)
plt.title('Error Test')
plt.show()