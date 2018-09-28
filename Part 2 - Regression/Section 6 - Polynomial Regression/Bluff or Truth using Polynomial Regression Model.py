# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:32:42 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create DataSet
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values

# Create Regression Model
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X,Y)

# Create Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
polyRegX = poly_reg.fit_transform(X)
poly_reg.fit(polyRegX,Y)
polynomialRegressor = LinearRegression()
polynomialRegressor.fit(polyRegX,Y)

#Visualize the Linear Regression dataset
plt.scatter(X,Y,color='red')
plt.plot(X,linearRegressor.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression dataset
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,polynomialRegressor.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.show()

#Predict the linear Regression Model
linearRegressor.predict(6.7)

#Predict the Polynomial Regression dataset
polynomialRegressor.predict(poly_reg.fit_transform(6.5))