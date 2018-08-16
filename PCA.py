import csv
import numpy as np
def loadDataset(filename):
    data=[]
    with open(filename, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        dataS = list(rows)
        for x in range(len(dataS)):
            temp=[]
            for y in range(34):
                temp.append(float(dataS[x][y]))
            data.append(temp)
    return data
    
data=loadDataset('C:\\Users\\1992n\Documents\\NIT Rourkela\\ML\\ionosphere.data.txt')
#data=[[1,2],[2,3],[3,2],[4,4],[5,4],[6,7],[7,6],[9,7]]
''' Find the sample mean '''
mean=np.ndarray.tolist(np.mean(data,axis=0))

'''Subtract sample mean from the data'''
for i in range(len(data)):
    data[i]=np.ndarray.tolist(np.subtract(data[i],mean))
    
'''Compute the scatter matrix'''
S=np.ndarray.tolist((len(data))*np.cov(np.array(data).T))

''' Compute eigen values and vectors'''
w,v=np.linalg.eig(S)

''' Storing the correspoding value and vector in non-increasing order '''
X=[]
for i in range(len(w)):
    temp=[w[i],np.ndarray.tolist(v[i])]
    X.append(temp)
   
X.sort(key=lambda x:x[0],reverse=True)


''' Calculating d for dimensional reduction '''
lambda_sum=0
for i in range(len(X)):
    lambda_sum+=X[i][0]
d=0
lam=0
for i in range(len(X)):
    if((lam*1.0/lambda_sum)>=0.9):
        break
    else:
        lam+=X[i][0]
        d+=1

''' Calculating req eigen vectors '''
eigenVec=[]
for i in range(d):
    eigenVec.append(X[i][1])
    
''' final required data set '''
Y=np.ndarray.tolist(np.matmul(np.array(data),np.array(eigenVec).T))
print "PCA dimension reduction reduces data set of "+str(len(data))+"x"+str(len(data[0]))+" to "+str(len(Y))+"x"+str(len(Y[0]))
