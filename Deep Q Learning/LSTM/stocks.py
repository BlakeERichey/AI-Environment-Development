# importing the requests library 
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

def save_json(obj, filename):
  '''save obj to filename. obj is expected to be a json format'''
  with open(filename, 'w+') as f:
    json.dump(obj, f)

def load_json(filename):
  '''returns dictionary with json data'''
  with open(filename, 'r') as f:
    obj = json.loads(f.read())
  
    return obj

req = False
if req:
  stock = 'MSFT'
  # api-endpoint 
#  URL = "http://maps.googleapis.com/maps/api/geocode/json"
  URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&apikey=T7XYX1RHMEOB36OS'
  # location given here 
#  location = "delhi technological university"
    
  # defining a params dict for the parameters to be sent to the API 
#  PARAMS = {'address':location} 
    
  # sending get request and saving the response as response object 
  r = requests.get(url = URL)
    
  # extracting data in json format 
  data = r.json()

  save_json(data["Time Series (Daily)"], 'msft.json')

df = pd.read_json('./msft.json')
y = []
i = 0
for date in df:
  if i < 40: #5 years
    print(date, df[date]['4. close'])
    y.append(df[date]['4. close'])
  i+=1
x = [i for i in range(len(y))]
y = y[::-1]
#plt.plot(x, y)
# plt.show()
# print()



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
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
#model.compile(loss="mae", optimizer="rmsprop", metrics=['mse'])
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mse'])
model.summary()

true_y = y
mean_adj = 130
plt.plot(x, y, label='Actual')
y = [val-mean_adj for val in y]
x_train = np.array([np.array([i]) for i in x])
y_train = np.array([np.array([j]) for j in y])
print(x_train.shape)
x_train = np.reshape(x_train, (len(x),1,1))
print(x_train.shape)
ckpt = ModelCheckpoint('./best_model.h5', monitor='loss', verbose=0, save_weights_only=True, save_best_only=True)
#model.fit(x_train, y_train, verbose=1, batch_size=4, epochs=1000, callbacks=[ckpt])
#model.save_weights('msft.h5')
model.load_weights('./best_model.h5')
x = []
y = []
for i in range(len(true_y)):
    x.append(i)
    guess = model.predict(np.array(x).reshape(i+1, 1, 1))[-1][0]
#    print(guess)
    y.append(guess + mean_adj)

#y = [i.tolist()[0] for i in y]
day = len(x)
guesses = model.predict(np.array([i for i in range(day+1)]).reshape(day+1, 1, 1))
print(guesses)
today = guesses[-2][0]
tomorrow = guesses[-1][0]
print('Todays close:', true_y[-1], 'Guess:', today+mean_adj)
print('Tomorrows guess:', tomorrow+mean_adj)
plt.plot(x, y, label='guess')
plt.legend(loc=3)
r2 = metrics.r2_score(true_y, y)
print('R:', pow(r2, .5))
plt.show()
#print(x[:3], model.predict(np.array([1,2,3,4]).reshape(4,1,1)))
