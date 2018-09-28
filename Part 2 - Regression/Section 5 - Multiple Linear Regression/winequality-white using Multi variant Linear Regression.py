# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:43:13 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create dataset
dataset = pd.read_csv('winequality-white.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Create training and test set
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state = 0)

# Create a Prediction model Using Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

# Create a prediction using training set
predictY = regressor.predict(testX)