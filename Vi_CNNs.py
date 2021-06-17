import tensorflow as tf
import keras
import numpy as np
import pandas as pd


## Models

def generate_network (lr = 3e-4, drop = 0.2, l1=0.001, l2 = 0.001):
  
  reg = keras.regularizers.l1_l2(l1 = l1, l2 = l2)
  #reg = None
  
  #filters = 64
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(64,kernel_regularizer = reg, kernel_size = 5, strides = 2, activation = 'relu', padding = "same",
                                input_shape = (312, 312, 3)))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.1))
  
  model.add(keras.layers.Conv2D(128, kernel_regularizer = reg,kernel_size = 3, activation="relu", padding = "same"))         
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Conv2D(128, kernel_regularizer = reg,kernel_size = 3, activation="relu", padding = "same"))          
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.1))
  
  model.add(keras.layers.Conv2D(256, kernel_regularizer = reg,kernel_size = 3, activation="relu", padding = "same"))          
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Conv2D(256,  kernel_regularizer = reg,kernel_size = 3, activation="relu", padding = "same"))        
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.1))
 
  model.add(keras.layers.Flatten())
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(64, kernel_regularizer = reg,activation = "relu"))
  model.add(keras.layers.Dropout(drop))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(16,kernel_regularizer = reg,activation = "relu"))
  model.add(keras.layers.Dropout(drop))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(3, activation = "softmax"))

  optimizer = keras.optimizers.Adam(learning_rate = lr)
  

  model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, metrics=['accuracy'])
                
  return model

#------------------------------------------

def generate_network_2 (lr = 3e-4, drop = 0.2, l2 = 0.001):
  
  reg = keras.regularizers.l1(l2)
  
  #filters = 64
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(64,kernel_size = 5, kernel_regularizer = reg, strides = 2, activation = 'relu', padding = "same",
                                input_shape = (312, 312, 3)))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.1))
  
  model.add(keras.layers.Conv2D(128,kernel_regularizer = reg, kernel_size = 3, activation="relu", padding = "same"))         
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Conv2D(128, kernel_size = 3, activation="relu", padding = "same"))          
  #model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.2))
  
  model.add(keras.layers.Conv2D(256, kernel_regularizer = reg,kernel_size = 3, activation="relu", padding = "same"))          
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Conv2D(256,  kernel_size = 3, activation="relu", padding = "same"))        
  #model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.MaxPooling2D((2,2), padding = "same"))
  model.add(keras.layers.normalization.BatchNormalization())
  #model.add(keras.layers.Dropout(0.3))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(64,  kernel_regularizer = reg,activation = "relu"))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(16,kernel_regularizer = reg,activation = "relu"))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(keras.layers.Dense(3, activation = "softmax"))

  # tf.keras.layers.Lambda(tf.nn.local_response_normalization)
  optimizer = keras.optimizers.Adam(learning_rate = lr)
  

  model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, metrics=['accuracy'])
                
  return model
#-----------------------------------------------
