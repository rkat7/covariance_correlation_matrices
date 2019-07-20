import math
import numpy as np
m=np.array(['f1','f2','f3','f4'])
n1=np.array([23,12,45,67,44])
n2=np.array([9,31,76,23,45])
n3=np.array([11,5,1,78,123])
n4=np.array([56,23,19,66,8])

n=np.vstack((n1,n2))
n=np.vstack((n,n3))
n=np.vstack((n,n4))

n=n.T
n=np.vstack((m,n))
#Here, n represents the m*n sized matrix containing both feature vectors and samples


def traverse(u):
    gg=np.array([])
    gg=n[:, u]
    gg=gg[1:]
    return(gg)
    #Now, gg is the particular feature vector we are gonna use in calculation of stats 
  

def variance(x):
    var=0
    mean=0    
    for i in range(len(x)):
        mean+=int(x[i])
    mean=mean/len(x)
    for i in range(len(x)):
        var+=(int(x[i])-mean)**2
    var=var/len(x)-1
    return(var)

def covariance(x,y):
    covar=0
    mean1=0
    mean2=0
    for i in range(len(x)):
        mean1+=int(x[i])
        mean2+=int(y[i])
    mean1=mean1/len(x)
    mean2=mean2/len(y)
    
    for i in range(len(x)):
        covar+=(int(x[i])-mean1)*(int(y[i])-mean2)
    covar=covar/len(x)-1
    return(covar)
    
cov_mat=np.full((len(m),len(m)),0)
cor_mat=np.full((len(m),len(m)),0)


for i in range(len(m)):
    for j in range(len(m)):
        if (i==j):
            cov_mat[i,j]=variance(traverse(i))
            cor_mat[i,j]=0 #Since covariance of a feature with itself is 0
        else:
            cov_mat[i,j]=covariance(traverse(i),traverse(j))
            cor_mat[i,j]=cov_mat[i,j]/math.sqrt(variance(traverse(i)))*math.sqrt(variance(traverse(j)))

print(cov_mat,"is the Covariance matrix")
print("\n")
print(cor_mat,"is the Correlation matrix")
