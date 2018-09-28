# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 02:16:23 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create dataset
dataset = pd.read_excel('Concrete_Data.xls')
X = dataset.iloc[:,0:7].values
Y = dataset.iloc[:,8:].values

# Create training set
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.20, random_state = 0)


# Create a prediction on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

#Create the Prediction
predictY = regressor.predict(testX)

# create Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1030,1)).astype(int), values= X, axis = 1)
x_OPT = X[:,[0,1,2,3,4,5]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()