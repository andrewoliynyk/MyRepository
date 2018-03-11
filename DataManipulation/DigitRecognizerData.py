import pandas as pd

# "Digit_Recognizer/train.csv"
# "Digit_Recognizer/test.csv"

def getDigitRecognizerData(pTrainPath, pTestPath):
    
    train = pd.read_csv("../Coursework/Data/" + pTrainPath)
    test = pd.read_csv("../Coursework/Data/" + pTestPath)
    
    X_train = (train.ix[:,1:].values).astype('float32')
    y_train = train.ix[:,0].values.astype('int32')
    X_test = test.values.astype('float32')

    return X_train, y_train, X_test

def getDigitRecognizerDataForTesting(pTrainPath, pTrainCount):
    
    data = pd.read_csv("../Coursework/Data/" + pTrainPath)
    
    X_train = (data.ix[:pTrainCount,1:].values).astype('float32')
    y_train = data.ix[:pTrainCount,0].values.astype('int32')
    X_test = (data.ix[pTrainCount:,1:].values).astype('float32')
    y_test = data.ix[pTrainCount:,0].values.astype('int32')
    
    return X_train, y_train, X_test, y_test

