# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:48:57 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create dataset for Multi variant regression
dataset = pd.read_csv('usedcars.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


#Create Dummy Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])
#labelencoder = LabelEncoder()
#X[:,2] = labelencoder.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#onehotencoder = OneHotEncoder(categorical_features = [5])
#X = onehotencoder.fit_transform(X).toarray()

# Avoid Dummy variable trap
X = X[:,1:]

#Create Linear Regression
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state = 0)

# Create a prediction table using training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

# Predict the data
predictY = regressor.predict(testX)

#Backtrack elimination
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((150,1)).astype(int), values = X, axis =1)
X_opt = X[:,[0,1,2,3,4]]
regressor_opt = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_opt.summary()


