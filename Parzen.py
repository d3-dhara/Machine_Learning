from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
def generate_points_parzen(points):
    a=-5
    b=5
    answer=[]
    p=[]
    i=0
    sigma1_sq=0.2
    sigma2_sq=0.2
    while(i<points):
        temp=a + (b-a) * np.random.rand(1)
        temp2=(1.0/3)*pow(2*3.14*sigma1_sq,0.5)*np.exp(-1.0*pow(temp[0],2)/(2*sigma1_sq))+(2.0/3)*pow(2*3.14*sigma2_sq,0.5)*np.exp(-1.0*pow(temp[0]-2,2)/(2*sigma2_sq))
        if(temp2>np.random.rand(1)):
            answer.append(np.ndarray.tolist(temp))
            p.append(temp2)
            i+=1
    return answer,p

points=1000
h=0.1    
result=generate_points_parzen(points)
plt.plot(result[0],result[1],'.r')
sorted(result[0])
def parzen(result,h):
    p=[]
    z=[]
    N=len(result[0])
    d=len(result[0][1])
    i=-5
    while(i<=5):
        s=0
        for j in range(N):
            diff=np.subtract(i,result[0][j])
            m=np.matmul(np.transpose(diff),diff)
            s+=np.exp(-m/(2*pow(h,2)))
        ans=s*(1/(N*pow(44/7,d/2)*pow(h,d)))
        p.append(ans)
        z.append(i)
        i=i+h
    return z,p
r=parzen(result,h)
plt.plot(r[0],r[1],'b')

