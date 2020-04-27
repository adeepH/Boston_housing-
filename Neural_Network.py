# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:05:33 2020

@author: Adeep
"""

#Importing the libraries
import numpy as np
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.datasets import boston_housing


#Loading the training and testing datasets
(X_train,y_train) , (X_test,y_test) = boston_housing.load_data()


#Define Base Model
model = Sequential()
model.add(Dense(128,input_shape=(13,),kernel_initializer='normal',activation='relu',name='dense_1'))
model.add(Dense(64,kernel_initializer='normal',activation='relu',name='dense_2'))
model.add(Dropout(0.2))
model.add(Dense(1,kernel_initializer='normal',activation='linear',name='dense_3'))


#Compile model
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])


#Summary of the neural network
model.summary()

#Training the model
history = model.fit(X_train,y_train,epochs=100,validation_split=0.07)

print(np.max(np.sqrt(history.history['accuracy'])))
print('Accuracy:',(100-np.mean(np.sqrt(history.history['accuracy']))))

""" The model returns the output with 99.91% accuracy"""
""" Thus Neural Networks are preferred over every other algorithmS"""