# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 03:02:33 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values

# Create Random forest Regression Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,Y)



#Visualize the regression model
gridX = np.arange(min(X),max(X),0.01)
gridX = gridX.reshape((len(gridX),1))
plt.scatter(X,Y, color= 'blue')
plt.plot(gridX, regressor.predict(gridX),color = 'red')
plt.title('Random Forest Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() 