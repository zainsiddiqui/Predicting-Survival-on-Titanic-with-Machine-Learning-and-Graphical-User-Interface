# 
# Preprocesses data for Machine Learning models by dropping unncessary data as well as filling in blank values.
#
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns
import re

###########################################################
# Processes Data
def preProcess(data):
  
# Drop 'PassengerId' from data set as data column is not relevant to survival probability
    data = data.drop(['PassengerId'], axis=1)

# Dropping cabin data column due to missing data
    deckType = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data['Cabin'] = data['Cabin'].fillna("U0")
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(deckType)
    data['Deck'] = data['Deck'].fillna(0)
    data['Deck'] = data['Deck'].astype(int)

    data = data.drop(['Cabin'], axis=1)
# Handling age data column missing values

    mean = data["Age"].mean()
    std = data["Age"].std()
    is_null = data["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
    try:
      rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    except:
      rand_age = 0

# fill NaN values in Age column with random values generated
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = data["Age"].astype(int)

    fillVal = 'S'
    data['Embarked'] = data['Embarked'].fillna(fillVal)

# Converting Fare from float to int
    data['Fare'] = data['Fare'].fillna(0)
    data['Fare'] = data['Fare'].astype(int)

# Converting genders into 0 or 1
    gender = {"male": 0, "female":1}
    data['Sex'] = data['Sex'].map(gender)

# Dropping Ticket from dataset as data is not useful
    data = data.drop(['Ticket'], axis=1)

# Convering Embarked feature into numeric data
    ports = {"S": 0, "C": 1, "Q": 2}
    data['Embarked'] = data['Embarked'].map(ports)
#Dropping Name data as not useful
    data = data.drop(['Name'], axis=1)

# Creating age group
    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 4
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 5
    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6
    data.loc[ data['Age'] > 66, 'Age'] = 6

# Creating Fare group
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare']   = 3
    data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare']   = 4
    data.loc[ data['Fare'] > 250, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)
# Adding new column for Age Class
    data['Age_Class']= data['Age']* data['Pclass']
    return data
######################################################################

