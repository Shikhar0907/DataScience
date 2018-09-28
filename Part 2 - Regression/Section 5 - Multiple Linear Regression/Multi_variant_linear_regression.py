# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:08:02 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#now create Dummy variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid dummy variable trap
X = X[:,1:]


# Creating training and test set
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state = 0)

# creating Linear Regression training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

# Creation the Prediction
predictY = regressor.predict(testX)

#Create Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis = 1)
x_OPT = X[:,[0,1,2,3,4,5]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()
x_OPT = X[:,[0,1,3,4,5]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()
x_OPT = X[:,[0,3,4,5]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()
x_OPT = X[:,[0,3,5]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()
x_OPT = X[:,[0,3]]
regressor_opt = sm.OLS(endog = Y,exog = x_OPT).fit()
regressor_opt.summary()