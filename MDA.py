import csv
import numpy as np
def loadDataset(filename,data=[],classlabel=[]):
    with open(filename, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        dataS = list(rows)
        for x in range(len(dataS)):
            temp=[]
            for y in range(1,14):
                temp.append(float(dataS[x][y]))
            data.append(temp)
            classlabel.append(dataS[x][0])
    return data
    
data=[]
classlabel=[]
loadDataset('C:\\Users\\1992n\Documents\\NIT Rourkela\\ML\\wine.data.txt',data,classlabel)
no_of_classlabel=sorted(set(classlabel))
total_data=[]

total_mean=np.mean(data,axis=0)

for i in no_of_classlabel:
    temp=[]
    for j in range(len(data)):        
        if classlabel[j][0]==i:
            temp.append(data[j])
    total_data.append(temp)
            
total_data=np.array(total_data)
np.shape(total_data)

mean=[]
Scatter=[]
for i in range(len(total_data)):
    class_x=[]
    for j in range(len(total_data[i])):
        class_x.append(total_data[i][j])
    mean.append(np.mean(class_x,axis=0))
    Scatter.append(len(class_x)*np.cov(np.array(class_x).T))

S_w=np.array(0)
for i in range(len(Scatter)):
    S_w=np.add(np.array(S_w),Scatter[i])
    
S_b=np.array(0)
for i in range(len(no_of_classlabel)):
    x=[np.subtract(mean[i],total_mean)]*np.transpose([np.subtract(mean[i],total_mean)])
    S_b=np.add(S_b,np.array(len(total_data[i])*x))

w,v=np.linalg.eig(np.matmul(np.linalg.inv(S_w),S_b))
X=[]
for i in range(len(w)):
    temp=[w[i],v[i]]
    X.append(temp)
   
X.sort(key=lambda x:x[0],reverse=True)


''' Calculating d for dimensional reduction '''
lambda_sum=0
for i in range(len(X)):
    lambda_sum+=X[i][0]
d=0
lam=0
for i in range(len(X)):
    if(((lam+0.0)/lambda_sum)>=0.9):
        d=i
        break
    else:
        lam+=X[i][0]
''' Calculating req eigen vectors '''
eigenVec=[]
for i in range(d):
    eigenVec.append(X[i][1])
   
Y=np.matmul(data,np.array(eigenVec).T)
