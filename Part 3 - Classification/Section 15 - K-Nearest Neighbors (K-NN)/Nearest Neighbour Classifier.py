# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:46:21 2018

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create data set
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

# Create Training set and Test set model
from sklearn.cross_validation import train_test_split
trainX, testX, trainY,testY = train_test_split(X,Y,test_size = 0.25 , random_state = 0)

# Create Scaling
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
trainX = scalerX.fit_transform(trainX)
testX = scalerX.fit_transform(testX)

#Fitting the N neighbor classifiers
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(trainX,trainY)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p= 2)
classifier.fit(trainX,trainY)

# predicting the test set
predY = classifier.predict(testX)

#Predicting the test set
#predY = classifier.predict(testX)

#Making the confusion matrix
#from sklearn.metrics import confusion_matrix
#conf = confusion_matrix(testY,predY)

#Visualizing the training set
from matplotlib.colors import ListedColormap
setX,setY = trainX,trainY
X1,X2 = np.meshgrid(np.arange(start = setX[:,0].min() - 1, stop = setX[:,0].max() + 1, step = 0.01),
                    np.arange(start = setX[:,1].min() - 1, stop = setX[:,1].max() + 1,step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(setY)):
    plt.scatter(setX[setY == j, 0], setX[setY == j,1],
                c = ListedColormap(('red','green'))(i), label = j)

plt.title('K-nearset Neightbors (training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualizing the test set
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

plt.title('K-nearset Neightbors (training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


