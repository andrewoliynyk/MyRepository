import numpy as np 
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import time
from DataManipulation.DigitRecognizerData import *
from DataManipulation.TitanicData import *
from Algorithms.kNN import apply_kNN
from Algorithms.LogReg import LogRegTitanicTest

# "Digit_Recognizer/Etalons_k=1_1000.csv"
# "Digit_Recognizer/Etalons_k=3_2000.csv"
# "Digit_Recognizer/train.csv"
# "Digit_Recognizer/test.csv"

def kNNDigitRecognizerTest():
    start = time.time()
    result = []
    X_train, y_train, X_test = getDigitRecognizerData("Digit_Recognizer/train.csv", "Digit_Recognizer/test.csv")
    for i in range(X_test.shape[0]):
        Answer = apply_kNN(X_test[i],X_train,y_train,3)
        result.append(Answer)
        if i % 100 == 0:
            print("Working ... %d %%" % (100*i/X_test.shape[0]))
    predictions=pd.DataFrame({"ImageId": list(range(1,len(result)+1)),
                         "Label": result})
    predictions = predictions.fillna(0)
    predictions = predictions.astype(str)
    predictions.to_csv("predictions3.csv", index=False, header=True)
    return (time.time() - start)

def kNNTitanicTest():
    start = time.time()
    result = []
    
    X_train, y_train, X_test = prepareTitanicData("Titanic/train.csv", "Titanic/test.csv")
    for i in range(X_test.shape[0]):
        Answer = apply_kNN(X_test[i],X_train,y_train,5)
        result.append(Answer)
        if i % 100 == 0:
            print("Working ... %d %%" % (100*i/X_test.shape[0]))
    predictions=pd.DataFrame({"PassengerId": list(range(892,1310)),
                         "Survived": result})
    predictions = predictions.fillna(0)
    predictions = predictions.astype(str)
    predictions.to_csv("TitanicPredictions3.csv", index=False, header=True)
    return (time.time() - start)

if __name__ == "__main__":
    print(LogRegTitanicTest())
