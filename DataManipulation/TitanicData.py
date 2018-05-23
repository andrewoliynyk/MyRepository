import numpy as np 
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt

def prepareTitanicData(pTrainPath, pTestPath):

    train = pd.read_csv("../Coursework/Data/" + pTrainPath)
    test = pd.read_csv("../Coursework/Data/" + pTestPath)

    full_data = [train, test]

    full_data[0] = full_data[0].drop(['Cabin'], axis=1)
    full_data[1] = full_data[1].drop(['Cabin'], axis=1)

    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    full_data[0] = full_data[0].drop(['Name'], axis=1)
    full_data[1] = full_data[1].drop(['Name'], axis=1)

    full_data[0] = full_data[0].drop(['Ticket'], axis=1)
    full_data[1] = full_data[1].drop(['Ticket'], axis=1)

    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    average_age_titanic   = full_data[0]["Age"].mean()
    std_age_titanic       = full_data[0]["Age"].std()
    count_nan_age_titanic = full_data[0]["Age"].isnull().sum()

    average_age_test   = full_data[1]["Age"].mean()
    std_age_test       = full_data[1]["Age"].std()
    count_nan_age_test = full_data[1]["Age"].isnull().sum()

    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
    rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

    full_data[0]["Age"][np.isnan(full_data[0]["Age"])] = rand_1
    full_data[1]["Age"][np.isnan(full_data[1]["Age"])] = rand_2

    full_data[0]['Age'] = full_data[0]['Age'].astype(int)
    full_data[1]['Age'] = full_data[1]['Age'].astype(int)

    full_data[1]["Fare"].fillna(full_data[1]["Fare"].median(), inplace=True)

    for dataset in full_data:
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int) 

    for dataset in full_data:
        dataset['EarlyTwentieth'] = 0
        dataset.loc[dataset['Age'] == 20, 'EarlyTwentieth'] = 1
        dataset.loc[dataset['Age'] == 21, 'EarlyTwentieth'] = 1 
        dataset.loc[dataset['Age'] == 22, 'EarlyTwentieth'] = 1 

    for dataset in full_data:
        dataset.loc[ dataset['Age'] <= 6, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 15), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 62), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
    
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    full_data[0] = full_data[0].drop(['SibSp'], axis=1)
    full_data[1] = full_data[1].drop(['SibSp'], axis=1)
    full_data[0] = full_data[0].drop(['Parch'], axis=1)
    full_data[1] = full_data[1].drop(['Parch'], axis=1)
    
    for dataset in full_data:
        dataset.loc[ dataset['FamilySize'] == 1, 'FamilySize'] 					    = 1
        dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'FamilySize'] = 2
        dataset.loc[dataset['FamilySize'] > 4, 'FamilySize'] = 3
    
    for dataset in full_data:    
        dataset.loc[ dataset['Fare'] <= 8, 'Fare'] 						= 0
        dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    
    for dataset in full_data:
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    Y = full_data[0]['Survived']
    X = full_data[0].drop(['Survived', 'PassengerId'], axis=1)

    X = X.values.astype('float32')
    Y = Y.values.astype('int64')
    X_test = full_data[1][['Pclass','Sex','Age','Fare','Embarked','Title','EarlyTwentieth','FamilySize']].values.astype('float32')

    return (X, Y, X_test)

#def getTitanicData(pTrainPath, pTestPath):
    
    #train, test = prepareTitanicData(pTrainPath, pTestPath)
    
    #X_train = (train.ix[:,0].values).astype('float32') ((train.ix[:,2].values).astype('float32'))
    #y_train = (train.ix[:,1].values).astype('int32')
    #X_test = (test.ix[:,0].values).astype('float32').append((test.ix[:,2].values).astype('float32'))
    #y_test = (test.ix[:,1].values).astype('int32')
    
    #print(X_train)

    #print(y_train)
    #return (X_train, y_train, X_test)


    #=============================================================

def prepareTitanicData1(pTrainPath, pTestPath):

    train = pd.read_csv("../Coursework/Data/" + pTrainPath)
    test = pd.read_csv("../Coursework/Data/" + pTestPath)

    full_data = [train, test]

    full_data[0] = full_data[0].drop(['Cabin'], axis=1)
    full_data[1] = full_data[1].drop(['Cabin'], axis=1)

    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    full_data[0] = full_data[0].drop(['Name'], axis=1)
    full_data[1] = full_data[1].drop(['Name'], axis=1)

    full_data[0] = full_data[0].drop(['Ticket'], axis=1)
    full_data[1] = full_data[1].drop(['Ticket'], axis=1)

    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    average_age_titanic   = full_data[0]["Age"].mean()
    std_age_titanic       = full_data[0]["Age"].std()
    count_nan_age_titanic = full_data[0]["Age"].isnull().sum()

    average_age_test   = full_data[1]["Age"].mean()
    std_age_test       = full_data[1]["Age"].std()
    count_nan_age_test = full_data[1]["Age"].isnull().sum()

    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
    rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

    full_data[0]["Age"][np.isnan(full_data[0]["Age"])] = rand_1
    full_data[1]["Age"][np.isnan(full_data[1]["Age"])] = rand_2

    full_data[0]['Age'] = full_data[0]['Age'].astype(int)
    full_data[1]['Age'] = full_data[1]['Age'].astype(int)

    full_data[1]["Fare"].fillna(full_data[1]["Fare"].median(), inplace=True)

    for dataset in full_data:
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int) 

    for dataset in full_data:
        dataset['EarlyTwentieth'] = 0
        dataset.loc[dataset['Age'] == 20, 'EarlyTwentieth'] = 1
        dataset.loc[dataset['Age'] == 21, 'EarlyTwentieth'] = 1 
        dataset.loc[dataset['Age'] == 22, 'EarlyTwentieth'] = 1 

    for dataset in full_data:
        dataset.loc[ dataset['Age'] <= 6, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 15), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 62), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
    
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    full_data[0] = full_data[0].drop(['SibSp'], axis=1)
    full_data[1] = full_data[1].drop(['SibSp'], axis=1)
    full_data[0] = full_data[0].drop(['Parch'], axis=1)
    full_data[1] = full_data[1].drop(['Parch'], axis=1)
    
    for dataset in full_data:
        dataset.loc[ dataset['FamilySize'] == 1, 'FamilySize'] 					    = 1
        dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'FamilySize'] = 2
        dataset.loc[dataset['FamilySize'] > 4, 'FamilySize'] = 3
    
    for dataset in full_data:    
        dataset.loc[ dataset['Fare'] <= 8, 'Fare'] 						= 0
        dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    
    for dataset in full_data:
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    Y = full_data[0]['Survived']
    X = full_data[0].drop(['Survived', 'PassengerId'], axis=1)

    X = X.values.astype('float32')
    Y = Y.values.astype('int64')

    X_test = full_data[1][['Pclass','Sex','Age','Fare','Embarked','Title','EarlyTwentieth','FamilySize']].values.astype('float32')

    X = np.array(list(zip(X,Y)))
    Y = np.array(list(zip(X,Y)))
    X_test = np.array(X_test)
    return (X, Y, X_test)