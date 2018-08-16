import csv
import numpy as np
class_b=[]
class_g=[]

def loadDataset(filename):
    with open(filename, 'rb') as csvfile:
        rows = csv.reader(csvfile)
        dataS = list(rows)
        for x in range(len(dataS)):
            temp=[]
            for y in range(34):
                temp.append(float(dataS[x][y]))
            if(dataS[x][34]=='b'):
                class_b.append(temp)
            else:
                class_g.append(temp)
    
loadDataset('C:\\Users\\1992n\Documents\\NIT Rourkela\\ML\\ionosphere.data.txt')

mean_b=np.mean(class_b,axis=0)
mean_g=np.mean(class_g,axis=0)

S_b=len(class_b)*np.cov(np.array(class_b).T)
S_g=len(class_g)*np.cov(np.array(class_g).T)

Within_class_scatter=np.add(S_b,S_g)

optimal_line_direction=np.matmul(np.linalg.pinv(Within_class_scatter),np.subtract(mean_b,mean_g))

Y_b=np.ndarray.tolist(np.matmul(optimal_line_direction.T,np.array(class_b).T))
Y_g=np.ndarray.tolist(np.matmul(optimal_line_direction.T,np.array(class_g).T))

print "FLDA dimension reduction reduces data set of 1st class "+str(len(class_b))+"x"+str(len(class_b[0]))+" to "+str(len(Y_b))+"x 1"
print "FLDA dimension reduction reduces data set of 2nd class "+str(len(class_g))+"x"+str(len(class_g[0]))+" to "+str(len(Y_g))+"x 1"

