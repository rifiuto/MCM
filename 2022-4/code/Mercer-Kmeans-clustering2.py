#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
hair_dryer_data=pd.read_excel('hair_dryer_data_1.xlsx')
microwave_data=pd.read_excel('microwave_data_1.xlsx')
pacifier_data=pd.read_excel('pacifier_data_1.xlsx')




from numpy import *
import matplotlib.pyplot as plt
#     
def loadDataSet(fileName):  #       tab                  
    dataMat = []              #               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)    #        float  
        dataMat.append(fltLine)
    return dataMat

#         
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #           

#         k (   k=4)    
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   #      n        k   
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-means     
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    '''
    :param dataSet:    lable      (        )
    :param k:       
    :param distMeans:           
    :param createCent:     k        
    :return: centroids        k    
            clusterAssment:                     
    '''
    m = shape(dataSet)[0]   #m=80,    
    clusterAssment = mat(zeros((m,2)))
    # clusterAssment                              
    centroids = createCent(dataSet, k)
    clusterChanged = True   #             
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  #                   
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  #    i      j         i   j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True  #                 
            clusterAssment[i,:] = minIndex,minDist**2   #    i             
        # print centroids
        for cent in range(k):   #        
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   #       cent    
            centroids[cent,:] = mean(ptsInClust, axis = 0)  #           
    return centroids, clusterAssment
# --------------------  ----------------------------------------------------
#         kmeans  




if __name__ == '__main__':
    datMat = h
    # print min(datMat[:,0])
    # print max(datMat[:,1])
    # print randCent(datMat,4)
    myCentroids,clustAssing = kMeans(datMat,3)
    print( myCentroids)
    # print clustAssing,len(clustAssing)

    plt.figure(1)
    x=array(datMat[:,0]).ravel()
    y=array(datMat[:,1]).ravel()
    plt.scatter(x,y, marker='o')
    xcent=array(myCentroids[:,0]).ravel()
    ycent=array(myCentroids[:,1]).ravel()
    plt.scatter( xcent, ycent, marker='x', color='r', s=50)
    plt.show()




def check_helpful(df):
    check_out=[]
    all_list=[]
    for index,row in df.iterrows():
        x=2*row['helpful_votes']-row['total_votes']
        all_list.append(1/(1+np.exp(-1*x)))
        if row['total_votes']!=0:
            if 1/(1+np.exp(-1*x))<0.5:
                check_out.append(index)
    print(check_out)
    print(all_list)
    return check_out,all_list




a,b=check_helpful(pacifier_data)




b=pd.DataFrame(b)
h=pd.concat([pacifier_data['star_rating'],b],axis=1)




h=h.values




import sys
np.set_printoptions(threshold=sys.maxsize)
print(h)




def sigmoid(x):
    #     sigmoid  
    return 1. / (1. + np.exp(-x))
 
 
def plot_sigmoid():
    # param:        
    x = np.arange(-8, 0, 0.2)
    x1 = np.arange(0, 8, 0.2)
    y = sigmoid(x)
    y1 = sigmoid(x1)
    plt.scatter(x, y)
    plt.scatter(x1, y1)
    plt.show()
 
 
if __name__ == '__main__':
    plot_sigmoid()






