import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

train = pd.read_csv("../Coursework/Data/Titanic/train.csv",sep=',')
test = pd.read_csv("../Coursework/Data/Titanic/test.csv",sep=',')

print(train.head())