import numpy as np
from DataManipulation.DigitRecognizerData import *
from Algorithms.kNN import apply_kNN

# "Digit_Recognizer/Etalons_k=1_1000.csv"
# "Digit_Recognizer/Etalons_k=3_2000.csv"
# "Digit_Recognizer/train.csv"
# "Digit_Recognizer/test.csv"

def DigitRecognizerTest():
    Counter = 0
    X_train, y_train, X_test, y_test = getDigitRecognizerDataForTesting("Digit_Recognizer/train.csv", 10000)
    for i in range(y_test.shape[0]):
        Answer = apply_kNN(X_test[i],X_train,y_train,3)
        if Answer == y_test[i]:
            Counter = Counter + 1
        if i % 100 == 0:
            print("Working ... %d %% Correct = %d %%" % (100*i/y_test.shape[0], 100*Counter/(i+1)))
    return (100*Counter / y_test.shape[0])

print("Correct = %d %%" % DigitRecognizerTest())
    


