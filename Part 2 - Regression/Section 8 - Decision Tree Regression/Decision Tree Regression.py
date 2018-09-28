# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:08:39 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values

#Create decision Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)


# predict the tree
predictY = regressor.predict(6.5)


#Create visualization of Decision tree
gridX = np.arange(min(X),max(X),0.1)
gridX = gridX.reshape((len(gridX),1))
plt.scatter(X,Y, color = 'red')
plt.plot(gridX,regressor.predict(gridX),color='blue')
plt.title('Decision Tree')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()