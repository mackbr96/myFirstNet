import numpy as np
import matplotlib.pyplot as plt

from random import random
from random import seed
from math import exp
import math

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 1000
        self.outputSize = 1
        self.hiddenSize = 10
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1) 
        self.z2 = self.sigmoid(self.z) 
        self.z3 = np.dot(self.z2, self.W2) 
        o = self.sigmoid(self.z3) 
        return o.astype(float)

    def sigmoid(self, s):
        return (2/(1+np.exp(-2*s)))-1

    def sigmoidPrime(self, s):
        return 1- np.square(self.sigmoid(s))

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


    def backward(self, X, y, o):
        self.o_error = np.subtract(y,o)
        self.o_delta = np.multiply(self.o_error,self.sigmoidPrime(o)) 

        self.z2_error = np.dot(self.o_delta, self.W2.T) 
        self.z2_delta = np.multiply(self.z2_error,self.sigmoidPrime(self.z2)) 

        self.W1 += np.dot(X.T, self.z2_delta) 
        self.W2 += np.dot(self.z2.T, self.o_delta) 


dataset = []
labels = []
with open("train.txt") as f:
    lines = f.readlines()
    for i in lines:
        dataset.append(i.split())

for i in range(0, len(dataset)):
    for x in range(0,len(dataset[i])):
        dataset[i][x] = float(dataset[i][x])
i = 0
lines = []
with open("label.txt") as f:
    lines = f.readlines()
    for line in lines:
        labels.append(line.split())
        i += 1
for i in range(0, len(labels)):
    for x in range(0,len(labels[i])):
        dataset[i][x] = float(labels[i][x])


data= np.array(dataset, dtype=np.float128)
label = np.array(labels, dtype=np.float128)


NN = Neural_Network()
for i in range(0,15): # trains the NN 1,000 times
    print ("Loss: \n" + str(np.mean(np.square(label - NN.forward(data))))) # mean sum squared loss
    print ("\n")
    NN.train(data, label)




