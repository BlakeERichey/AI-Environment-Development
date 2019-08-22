# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 03:32:19 2019

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
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

import sklearn.metrics as metrics
    
#model = Sequential()
#model.add(Dense(16, input_shape=(1,)))
#model.add(Dense(32))
#model.add(Dense(1, activation = 'linear'))
#model.compile(optimizer = Adam(lr=0.001), loss = 'mae')
    
model = Sequential()
model.add(LSTM(64, input_shape=(1,1), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
#model.compile(loss="mae", optimizer="rmsprop", metrics=['mse'])
model.compile(loss="mae", optimizer="adam", metrics=['mse'])
model.summary()

x = [i for i in range(40)]
y = [random.randint(5, 10) for x in range(20)]
y += [random.randint(10, 15) for x in range(10)]
y += [random.randint(0, 5) for x in range(10)]

true_y = y

plt.plot(x, y, label='Actual')
x_train = np.array([np.array([i]) for i in x])
y_train = np.array([np.array([j]) for j in y])
print(x_train.shape)
x_train = np.reshape(x_train, (40,1,1))
print(x_train.shape)
early = ModelCheckpoint('./best_model.h5', monitor='loss', verbose=0, save_weights_only=True, save_best_only=True)
model.fit(x_train, y_train, verbose=1, epochs=5000, callbacks=[early])
model.load_weights('best_model.h5')
x = []
y = []
for i in range(40):
    x.append(i)
    guess = model.predict(np.array(x).reshape(i+1, 1, 1))
#    print(guess)
    y.append(guess[-1][0])

#y = [i.tolist()[0] for i in y]
#print(model.predict(np.array([i for i in range(43)]).reshape(43, 1, 1)))
plt.plot(x, y, label='guess')
plt.legend(loc=3)
r2 = metrics.r2_score(true_y, y)
print('R:', pow(r2, .5))
#print(x[:3], model.predict(np.array([1,2,3,4]).reshape(4,1,1)))