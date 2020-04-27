# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:09:04 2020

@author: Adeep
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb 
from keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error


#loading the training and testing datasets
(X_train,y_train),(X_test,y_test)  = boston_housing.load_data()

#Calling the model
xg_reg = xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,
                          learning_rate=0.1,max_depth = 5, alpha=5,
                          n_estimators = 10)
#Fitting the model
xg_reg.fit(X_train, y_train)

#Predicting the model 
y_pred = xg_reg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2 = r2_score(y_train, y_pred)

print("The model performance for training  set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Evaluating the model
y_pred =xg_reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#Visualise boosting Trees and Feature Importance
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
data_matrix = xgb.DMatrix(data = X_train,label = y_train)
xg_reg = xgb.train(params, dtrain=data_matrix,
                   num_boost_round=10)


#Plotting the second Tree using the matplotlib library
xgb.plot_tree(xg_reg, num_trees=2)
plt.rcParams['figure.figsize']=[30,30]
plt.show()

#Importance of each feature column in the original dataset within the model
xgb.plot_importance(xg_reg,color='red')
plt.rcParams['figure.figsize']=[30,30]
plt.show()
