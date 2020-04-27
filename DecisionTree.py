# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:52:29 2020

@author: Adeep
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeRegressor
from keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score


#loading the training and testing datasets
(X_train,y_train),(X_test,y_test)  = boston_housing.load_data()

tree = DecisionTreeRegressor()

# Fitting  the mode
tree.fit(X_train, y_train)

#Predicting the model
y_pred = tree.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2 = r2_score(y_train, y_pred)

print("The model performance for training  set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Evaluating the model
y_pred =tree.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Visualising the results
sklearn.tree.plot_tree(tree,max_depth=2)