# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:52:25 2017

"""
# One vs All logistic regression for Iris data set
import math
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

# hypothesis function outputs values of 0 or 1

def createData():
    inFile=open('irisdata.txt', 'r')
    Xiris=[];Yiris=[];thetaXi=[];
    theta=np.matrix([[1],[0],[1],[0],[1]],dtype='float64')
    thetaXi.append(theta.copy());thetaXi.append(theta.copy());thetaXi.append(theta.copy());
    for i,line in enumerate(inFile.readlines()):
        Xiris.append(line.strip().split(','))
        Yiris.append(Xiris[i].pop())
        
    #trimming the data and converting to np.matrix
    Xiris=Xiris[:-1];Yiris=Yiris[:-1];
    
    # normalizing X
    Xiris=np.matrix(Xiris,dtype='float64')
    featRange=(Xiris.max(axis=0)-Xiris.min(axis=0))
    avg=Xiris.mean(axis=0)
    Xiris=(Xiris-avg)/featRange
    m=np.shape(Xiris)[0]
    
    # adding the X0's as 1
    Xiris=np.hstack((np.matrix(np.ones((m,1))),Xiris))
     
    currentLabel=Yiris[0];count=0
    
    # converting the names to classes with indexes 0,1,2,...
    for i,n in enumerate(Yiris):
        if Yiris[i]==currentLabel:
            Yiris[i]=count;
        else: 
            currentLabel=Yiris[i]
            count+=1;Yiris[i]=count;
            
    Yiris=np.matrix(Yiris,dtype='float64');Yiris=Yiris.T.copy()
    return Xiris.copy(),Yiris.copy(),theta.copy(),avg,featRange,thetaXi


# Create Y vectors for each of the 3 classifications of 0s and 1s
def createY():
    
    YirisC=[]
    
    for i in range(3):
        YirisC.append(np.matrix((Yiris==i),dtype='float64'))
    return YirisC

def sigmoid(z):
    s=1/(1+np.exp(-z));
    return s


def computeCost(Y,h):
    J=-(1/m)*(np.sum(np.multiply((Y),np.log(h)))+np.sum(np.multiply((1-Y),np.log(1-h))))+(lmbda/(2*m))*np.sum(np.multiply(theta[1:],theta[1:]))
    return J.copy()


def GD(C,alpha):
    
    thetaXi[C]+=(alpha/m)*(Xiris.T*(YirisC[C]-sigmoid(Xiris*thetaXi[C])))+(lmbda*thetaXi[C]/m)
    J=computeCost(YirisC[C],sigmoid(Xiris*thetaXi[C]))
    return thetaXi[C].copy(),J.copy()

    
def learn(alpha1,alpha2,alpha3):
    J0data=[];J1data=[];J2data=[];accData=[]
    for i in range(_iter):

        thetaXi[0],J0=GD(0, alpha1)
        thetaXi[1],J1=GD(1, alpha2)
        thetaXi[2],J2=GD(2, alpha3)
        #comment out the 4 lines below for better performance
        J0data.append(J0)
        J1data.append(J1)
        J2data.append(J2)
        accData.append(predict(thetaXi)[1])
    print(f'\nTheta for 0s is {thetaXi[0].T}---- The cost function J is {J0}')
    print(f'Theta for 0s is {thetaXi[1].T}---- The cost function J is {J1}')
    print(f'Theta for 0s is {thetaXi[2].T}---- The cost function J is {J2}')
    return thetaXi,[J0data,J1data,J2data],accData

def predict(thetaXi):
    probability=np.zeros((len(Xiris),3));probability+=7;
    for X in range(len(Xiris)):
        probability[X][0]=sigmoid(Xiris[X]*thetaXi[0])  
        probability[X][1]=sigmoid(Xiris[X]*thetaXi[1])
        probability[X][2]=sigmoid(Xiris[X]*thetaXi[2])
    
    predictions=np.argmax(probability,1)
    
    predictions=np.matrix(predictions);predictions=predictions.T;
    bpred=(Yiris==predictions).astype(int)
    accuracy=int(sum(bpred))/len(bpred)
    return probability.copy(),accuracy
    


#takes in a feature vector X in the form of a list

def predictNew(X):
    X=np.matrix(X)
    X=(X-avg)/featRange
    X=np.hstack((np.matrix(np.ones((1,1))),X))
    probability=np.zeros((3,1))
    print(probability)
    for i in range(3):
        probability[i]=sigmoid(X*thetaXi[i])
    prediction=np.argmax(probability,1)
    
    return prediction,probability
        
    

# SCRIPT
#Comment out the 4 lines specified in the learn() function for better performance(without plots)
# hyperparameters
#alpha are argument in the learn function
    

lmbda=0
_iter= 30 # number of iterations of gradient descent

Xiris,Yiris,theta,avg,featRange,thetaXi=createData()
m=len(Xiris)
YirisC=createY()
#(100,.1,50)
thetaXi,Jdata,accData=learn(100,.1,50)
probabilities,accuracy=predict(thetaXi)
#plots the accuracy
accPlot=plt.figure()
plt.plot(accData,label='Training Accuracy')
accPlot.legend()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')

#plots the cost functions over iterations for each of the 3 classifiers
costPlot=plt.figure()
plt.plot(Jdata[0],label='classifier 0')
plt.plot(Jdata[1],label='classifier 1')
plt.plot(Jdata[2],label='classifier 2')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
costPlot.legend()
plt.ylim((0,2))
plt.show()



print(f'Training accuracy is {accuracy}')

