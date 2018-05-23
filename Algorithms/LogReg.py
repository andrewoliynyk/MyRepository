import numpy as np 
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import time
from DataManipulation.DigitRecognizerData import *
from DataManipulation.TitanicData import *
from Algorithms.kNN import apply_kNN

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def GA(DataX, ClassesY):
    mData = np.mat(DataX)  
    vClasses = np.mat(ClassesY).T
    m, n = np.shape(mData)
    alpha = 0.001
    loopLimit = 300
    w = np.ones((n, 1))
    for k in range(loopLimit): 
        w = w + alpha * mData.T*(vClasses - sigmoid(mData*w))
    return w

def SGA(mData, ClassesY):
    m, n = np.shape(mData)
    alpha = 0.01
    w = np.ones(n)  
    for i in range(m):
        h = sigmoid(sum(mData[i]*w))
        error = ClassesY[i] - h
        w = w + alpha * error * mData[i]
    return w

def apply_LogRegPred(X, w):
    if sigmoid(sum(X*w)) > 0.5:
        return 1
    else:
        return 0

def SGAMod(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def LogRegTitanicTest():
    start = time.time()
    result = []
    X_train, y_train, X_test = prepareTitanicData("Titanic/train.csv", "Titanic/test.csv")
    w = SGAMod(X_train, y_train, 300)
    for i in range(X_test.shape[0]):
        Answer = apply_LogRegPred(X_test[i],w)
        result.append(int(Answer))
        if i % 100 == 0:
            print("Working ... %d %%" % (100*i/X_test.shape[0]))
    predictions=pd.DataFrame({"PassengerId": list(range(892,1310)),
                         "Survived": result})
    predictions = predictions.fillna(0)
    predictions = predictions.astype(str)
    predictions.to_csv("TitanicPredictionsLogReg.csv", index=False, header=True)
    return (time.time() - start)



