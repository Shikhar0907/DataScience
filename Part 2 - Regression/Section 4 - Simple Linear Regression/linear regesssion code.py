# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:48:53 2018

@author: as
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
trainX, testX, trainY, testY = train_test_split(X,Y,test_size=1/3, random_state=0)

#feature Scalliing
"""from sklearn.preprocessing import StandardScaler

scX = StandardScaler()
trainX = scX.fit_transform(trainX)
testX = scX.transform(testX)"""

# fit Simple Linear Regression Model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

# Predicting the Test set results
predictY = regressor.predict(testX)

# plot the Regression graph on training set
plt.scatter(trainX,trainY,color='red')
plt.plot(trainX,regressor.predict(trainX),color='blue')
plt.title("Predicted Salary vs Original Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# plot the regression graph on test set
plt.scatter(testX,testY,color='red')
plt.plot(trainX,regressor.predict(trainX),color='blue')
plt.title("Predicted Salary vs Original Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

