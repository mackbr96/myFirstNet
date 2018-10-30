import numpy as np
import matplotlib.pyplot as plt

from random import random
from random import seed
from math import exp

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o 

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))


    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


    def backward(self, X, y, o):
        # backward propgate through the network
        print(type(o))
        print(type(y))
        self.o_error = np.subtract(y, o) # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights


dataset = []
labels = []
with open("2dPoints.txt") as f:
    lines = f.readlines()
    for i in lines:
        dataset.append(i.split())

for i in range(0, len(dataset)):
    for x in range(0,len(dataset[i])):
        dataset[i][x] = float(dataset[i][x])
i = 0
lines = []
with open("2dLabels.txt") as f:
    lines = f.readlines()
    for line in lines:
        #dataset[i].append(float(line))
        labels.append(line.split())
        i += 1
for i in range(0, len(labels)):
    for x in range(0,len(labels[i])):
        dataset[i][x] = float(labels[i][x])


data= np.array(dataset)
label = np.array(labels)

data = data/np.argmax(data, axis=0)
label = label


NN = Neural_Network()
for i in range(0,3): # trains the NN 1,000 times
    #print ("Input: \n" + str(data) )
    #print ("Actual Output: \n" + str(label) )
    #print ("Predicted Output: \n" + str(NN.forward(data))) 
    #print ("Loss: \n" + str(np.mean(np.square(label - NN.forward(data))))) # mean sum squared loss
    #print ("\n")
    NN.train(data, label)




