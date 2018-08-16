import csv
from random import random
from math import exp

def loadDataset(filename, ratio, trainingS=[] , testS=[]):
    with open(filename, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        dataS = list(rows)
        for x in range(len(dataS)-1):
            for y in range(4):
                dataS[x][y] = float(dataS[x][y])
            if dataS[x][-1] == 'Iris-setosa':
                dataS[x][-1] =[1,0,0]
            elif dataS[x][-1] == 'Iris-versicolor':
                dataS[x][-1] =[0,1,0]
            else:
                dataS[x][-1] =[0,0,1]           
            if random() < ratio:
                trainingS.append(dataS[x])
            else:
                testS.append(dataS[x])  
                
trainingSet=[]
testSet=[]

ratio = 0.67
loadDataset('C:\\Users\\1992n\\Documents\\NIT Rourkela\\ML\\iris.data.txt', ratio, trainingSet, testSet)
        
def initialize_network(inputs, n_hidden, outputs):
	network = list()
	hidden_layer = [{'weights':[0 for i in range(inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[0 for i in range(n_hidden + 1)]} for i in range(outputs)]
	network.append(output_layer)
	return network
 
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def sigmoid(activation):
	return 1.0 / (1.0 + exp(-activation))
 
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        temp_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            temp_inputs.append(neuron['output'])
        inputs = temp_inputs
    return inputs
  
def sigmoid_derivative(output):
    return output * (1.0 - output) 
 
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['sensitivity'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['sensitivity'] = errors[j] * sigmoid_derivative(neuron['output'])
   
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['sensitivity'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['sensitivity']
 
def train_network(network, train, l_rate, n_epoch, outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = row[-1]
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

inputs = len(trainingSet[0]) - 1
outputs = 3
network = initialize_network(inputs, 50, outputs)
train_network(network, trainingSet, 0.1, 500, outputs)

def testing(network,test_row,outputs):
    output=forward_propagate(network,test_row)
    answer=[0 for i in range(outputs)]
    answer[output.index(max(output))]=1
    return answer

matching=0
total=0

for i in range(len(testSet)):    
    if testSet[i][-1]==testing(network,testSet[i],outputs):
        matching+=1
    total+=1
    
print("Accuracy is:" +str((matching*1.0/total)*100))