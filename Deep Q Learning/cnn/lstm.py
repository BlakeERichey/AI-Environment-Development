# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:58:53 2019

@author: Blake
"""

import gym
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
from   tensorflow.keras.optimizers import Adam
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.python.client    import device_lib
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM, TimeDistributed
    
model = Sequential()

model.add(TimeDistributed(Conv2D(64, kernel_size=2, activation='relu'), input_shape=(24, None, 28, 1)))

model.add(TimeDistributed(MaxPooling2D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, stateful=True, return_sequences=True))
model.add(LSTM(10, stateful=True))
model.add(Dense(10))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()