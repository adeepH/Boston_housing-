# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:42:01 2020

@author: Adeep
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,r2_score


#loading the training and testing datasets
(X_train,y_train),(X_test,y_test)  = boston_housing.load_data()

Regressor = LinearRegression()

#fitting the model
Regressor.fit(X_train,y_train)

#Predicting the model 
y_pred = Regressor.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2 = r2_score(y_train, y_pred)

print("The model performance for training  set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Evaluating the model
y_pred = Regressor.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Plotting the scatter plot for the same model
plt.scatter(y_test,y_pred,color='blue')
plt.show()
