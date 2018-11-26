import numpy as np
np.random.seed(1)

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 1000
        self.hiddenSize = 10
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #Input weights
        self.W2 = np.random.randn(self.hiddenSize, 1) #Hidden layor weights
        #self.W2 = np.array([x / 20 for x in self.W2])
        #self.W1 = np.array([x / 20 for x in self.W1])

    def sigmoid(self, s):
        return (2/(1+np.exp(-2*s)))-1

    def sigmoidPrime(self, s):
        return 1- np.square(self.sigmoid(s))

    def forward(self, inputs):
        self.z = np.dot(inputs, self.W1)
        self.y1 = self.sigmoid(self.z)
        self.z2 = np.dot(self.y1, self.W2)
        output = self.sigmoid(self.z2)
        return output.astype(float)

    def backward(self, inputs, labels, output):
        self.outPut_error = np.subtract(labels,output)
        self.outPut_delta = np.multiply(self.outPut_error,self.sigmoidPrime(output))

        self.z2_error = np.dot(self.outPut_delta, self.W2.T)
        self.z2_delta = np.multiply(self.z2_error,self.sigmoidPrime(self.y1))

        self.W1 += np.dot(inputs.T, self.z2_delta)
        self.W2 += np.dot(self.y1.T, self.outPut_delta)

        #self.W1 = np.subtract(self.W1, (np.multiply(self.W1, self.z2_delta))
        #self.W2 = np.subtract(self.W2, self.outPut_delta)

    def train (self, data, labels):
        output = self.forward(data)
        self.backward(data, labels, output)

def parse(trainFile, labelFile):
    dataset = []
    labels = []
    with open(trainFile) as f:
        lines = f.readlines()
        for i in lines:
            dataset.append(i.split())

    lines = []
    with open(labelFile) as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.split())

    data= np.array(dataset, dtype=np.float64)
    label = np.array(labels, dtype=np.float64)
    return data, label

def testParse(trainFile, labelFile):
    dataset = []
    labels = []
    with open(trainFile) as f:
        lines = f.readlines()
        for i in lines:
            dataset.append(i.split())

    lines = []
    with open(labelFile) as f:
        lines = f.readlines()
        lines = lines[0][1:]
        lines = lines[:-1]
        lines = [x.strip() for x in lines.split(',')]
        for line in lines:
            labels.append(line)

    data = np.array(dataset, dtype=np.float64)
    label = np.array(labels, dtype=np.float64)
    return data, label


data, label = parse("train.txt", "label.txt")

NN = Neural_Network()

for i in range(0,20):
    #print(np.array(list(zip( label, NN.forward(data)))))
    print("Epoch " + str(i + 1))
    print ("Loss: " + str(np.mean(np.square(label - NN.forward(data)))))
    print("\n")
    NN.train(data, label)

right = 0
wrong = 0

data, label = testParse("a2-test-data.txt", "a2-test-label.txt")

predictions = NN.forward(data)

for i in range(0, len(label)):
    if label[i] == predictions[i]:
        right += 1
    else:
        wrong += 1
#print(np.array(list(zip( label, NN.forward(data)))))
print("Right: " + str(right))
print("Wrong: " + str(wrong))

np.savetxt("W1.txt", NN.W1)
np.savetxt("W2.txt", NN.W2)
