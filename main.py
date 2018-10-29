from random import random
from math import exp




def createNetwork(nInputs, nHidden):
    network = []
    hiddenLayor = [{'weight' : [ random() for i in range(nInputs + 1)]} for i in range(nHidden)]
    outPut = [{'weight' : random() for i in range(nHidden + 1)}]
    network.append(hiddenLayor)
    network.append(outPut)

    return network

def transfer(x):
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)

def transfer_derivative(output):
	return output * (1.0 - output)

def activate(weights, inputs):
    if type(weights) == float:
        weights = [weights]
    summation = float(weights[-1])
    for i in range(len(weights)-1):
        summation += weights[i] * float(inputs[i])
    

    return summation

def forwardPropogate(line, network):
    inputs = line
    for layor in network:
        new_inputs = []
        for neuron in layor:
            act = activate(neuron['weight'], inputs)
            neuron['output'] = transfer(act)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backPropogate(network, expected):
    for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weight'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']



def trainNetwork(network, data, lables):
    for line in data:
        outPut = forwardPropogate(line, network)
        expected = labels[i]
        backPropogate(network, labels)
        update_weights(network, line, .5)
        i += 1



with open("train.txt") as f:
        lines = f.readlines()
        data = []
        for i in lines:
            xx = [float(x) for x in i.split()]
            data.append(xx)
with open("labels.txt") as f:
    labels = []
    lines = f.readlines()
    for i in lines:
        labels.append(float(i))


print(labels)
network = createNetwork(len(data[0]) - 1, 10)
trainNetwork(network, data, labels)



