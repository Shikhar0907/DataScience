# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:44:11 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create Data Set
dataset = pd.read_csv('Swedish Auto Insurance Dataset.csv')
X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,1:].values

#Create a traing and test set
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size=0.2,random_state = 0)

# Now train the data and make the predictions set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

#Now predict using test set
predictY = regressor.predict(testX)

#Plot the Linear regression Graph
plt.scatter(testX,testY,color='blue')
plt.plot(trainX ,regressor.predict(trainX), color='red')
plt.title('Swedish Auto Insurance')
plt.xlabel('Number of claims')
plt.ylabel('Total payment for all claims in thousands of Swedish Kronor')
plt.show()