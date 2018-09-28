# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:45:34 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create Data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values


#Feature Sccalling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
X = scX.fit_transform(X)
Y = scY.fit_transform(Y)

# Create SVR Regression Model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

# Predict the training set
y_pred = scY.inverse_transform(regressor.predict(scX.transform(np.array([[6.5]]))))

# Visualize the SVR regression result
plt.scatter(X,Y, color = 'red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title('SVR Visualization Tabel')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


