import pandas as pd
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> kNN

# "Digit_Recognizer/train.csv"
# "Digit_Recognizer/test.csv"

def getDigitRecognizerData(pTrainPath, pTestPath):
    
    train = pd.read_csv("../Coursework/Data/" + pTrainPath)
    test = pd.read_csv("../Coursework/Data/" + pTestPath)
    
    X_train = (train.ix[:,1:].values).astype('float32')
<<<<<<< HEAD
    y_train = (train.ix[:,0].values).astype('int32')
    X_test = test.values.astype('float32')

    X_train = X_train / np.tile(255,(X_train.shape))
    X_test = X_test / np.tile(255,(X_test.shape))

    return X_train, y_train, X_test

def getDigitRecognizerDataEtalons(pTrainPath, pTestPath):
    
    train = pd.read_csv("../Coursework/Data/" + pTrainPath)
    test = pd.read_csv("../Coursework/Data/" + pTestPath)
    
    X_train = (train.ix[:,1:].values).astype('float32')
    y_train = (train.ix[:,0].values).astype('int32')
    X_test = (test.ix[:,1:].values).astype('float32')
    y_test = (test.ix[:,0].values).astype('int32')
    
    X_train = X_train / np.tile(255,(X_train.shape))
    X_test = X_test / np.tile(255,(X_test.shape))

    return X_train, y_train, X_test, y_test

=======
    y_train = train.ix[:,0].values.astype('int32')
    X_test = test.values.astype('float32')

    return X_train, y_train, X_test

>>>>>>> kNN
def getDigitRecognizerDataForTesting(pTrainPath, pTrainCount):
    
    data = pd.read_csv("../Coursework/Data/" + pTrainPath)
    
    X_train = (data.ix[:pTrainCount,1:].values).astype('float32')
<<<<<<< HEAD
    y_train = (data.ix[:pTrainCount,0].values).astype('int32')
    X_test = (data.ix[pTrainCount:,1:].values).astype('float32')
    y_test = (data.ix[pTrainCount:,0].values).astype('int32')

    X_train = X_train / np.tile(255,(X_train.shape))
    X_test = X_test / np.tile(255,(X_test.shape))

=======
    y_train = data.ix[:pTrainCount,0].values.astype('int32')
    X_test = (data.ix[pTrainCount:,1:].values).astype('float32')
    y_test = data.ix[pTrainCount:,0].values.astype('int32')
    
>>>>>>> kNN
    return X_train, y_train, X_test, y_test

