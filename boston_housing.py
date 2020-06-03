#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:29:49 2020

@author: ironman
"""
import numpy as np
#importing boston houses dataset
from keras.datasets import boston_housing
(train_data,train_targets) , (test_data,test_target) = boston_housing.load_data()
print(train_data.shape)
print(train_data.ndim)
print(test_data.shape)
print(test_data.ndim)

# Normalizing the data
mean = train_data.mean(axis = 0)
train_data = train_data-mean
std = train_data.std(axis = 0)
test_data = test_data - mean # calculate mean taken out from training data compulsorily, never calculate any normalization parameter from test data
test_data = test_data/std

#importing layers and models
from keras import layers
from keras import models

#Defining function to build model as required
def build_model():
    #Initialization of weights and biases
    model = models.Sequential()
    #first hidden layer
    model.add(layers.Dense(64, activation='relu',input_shape=((train_data.shape[1],))))
    #second hidden layer
    model.add(layers.Dense(64,activation='relu'))
    #output contains 1 neuron only so as to predict price
    model.add(layers.Dense(1))
    
    #Compiling the model
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    #Return model
    return model

#applying k fold cross validation
k=4
num_val_samples = len(train_data)//k
#num_val_samples = (num_val_samples)

#Saving the validation log at each fold
num_epochs = 500
all_mae_histories = []
for i in range(k):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()# shows after 80-90 epochs the model starts to overfit.   

x = np.array([100])
arr = x - mae_history
mean = arr.mean()
variance = arr.std()


 
