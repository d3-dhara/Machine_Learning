from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

Y_error=[]
X_epoch=[]

def hardlim(x):
    if x>=0:
        return 1.0
    else:
        return 0.0
 
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
         activation += weights[i + 1] * row[i]
    return hardlim(activation)
 
def train_weights(train, l_rate, n_epoch):
    weights = [np.random.rand(1)[0] for i in range(len(train[0]))]    
    for epoch in range(n_epoch):
        temp_error=0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            temp_error+=error**2
        X_epoch.append(epoch)
        Y_error.append(temp_error)
    return weights
 
def perceptron(training,test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(training, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)

iris = datasets.load_iris()

X=iris.data
Y=iris.target

data=[]
for i in range(len(X)):
    temp=[]
    if(Y[i]==0 or Y[i]==1):
        for j in range(len(X[0])):
            temp.append(X[i][j])
        temp.append(Y[i])
        data.append(temp)

test=[]
training=[]

for i in range(len(data)):
    if (np.random.rand(1)<0.66):
        training.append(data[i])
    else:
        test.append(data[i])
        
predictions=perceptron(training,test,0.1,10)
matched=0
total=0
for i in range(len(predictions)):
    if(test[i][-1]==predictions[i]):
        matched+=1
    total+=1
print("Accuracy is :"+str(((matched*1.0)/total)*100)+"%")
plt.plot(X_epoch,Y_error)