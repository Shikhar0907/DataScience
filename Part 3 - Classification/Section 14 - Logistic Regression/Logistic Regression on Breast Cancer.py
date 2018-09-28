# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:30:16 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create the Data set
dataset = pd.read_csv('breast-cancer-wisconsin.data.csv')
X = dataset.iloc[:,[1,2]].values
Y = dataset.iloc[:,-1].values

# Create training set
from sklearn.cross_validation import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.30, random_state = 0)

# Scalling the training set
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
trainX = scalerX.fit_transform(trainX)
testX = scalerX.fit_transform(testX)

# create prediction set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(trainX,trainY)

# predicting the training set
predictY = classifier.predict(testX)

#Creating the confucion metrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(testY,predictY)

# Visualization of the training set
from matplotlib.colors import ListedColormap
setX,setY = testX,testY
X1,X2 = np.meshgrid(np.arange(start = setX[:,0].min() - 1, stop = setX[:,0].max() + 1, step = 0.01),
                    np.arange(start = setX[:,1].min() - 1, stop = setX[:,1].max() + 1,step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(setY)):
    plt.scatter(setX[setY == j, 0], setX[setY == j,1],
                c = ListedColormap(('red','green'))(i), label = j)

plt.title('logistic regression (training set)')
plt.xlabel('Thickness')
plt.ylabel('Cell Size')
plt.legend()
plt.show()









