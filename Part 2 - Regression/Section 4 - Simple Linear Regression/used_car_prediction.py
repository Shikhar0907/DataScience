# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:45:55 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get the data
used_car_dataset = pd.read_csv('usedcars.csv')
X = used_car_dataset.iloc[:,:1].values
Y = used_car_dataset.iloc[:,3].values

#split the dataset in training and test data
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=1/3,random_state=0)

# now get the linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)

#now predicct the data
predictY = regressor.predict(testX)

#now plot the data on the graph
plt.scatter(trainX,trainY,color='red')
plt.plot(trainX,regressor.predict(trainX),color='blue')
plt.title('Car price Prediction')
plt.xlabel('Years')
plt.ylabel('Price')
plt.show()





