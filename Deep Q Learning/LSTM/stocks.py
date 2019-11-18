''' Gets closing prices for stocks and creates LSTM model to fit to the correlation'''

import gym
import json
import requests
import os, datetime, random
import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt
import sklearn.metrics   as metrics
from   tensorflow.keras.optimizers import Adam
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.python.client    import device_lib
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM


def save_json(obj, filename):
  '''save obj to filename. obj is expected to be a json format'''
  with open(filename, 'w+') as f:
    json.dump(obj, f)

def load_json(filename):
  '''returns dictionary with json data'''
  with open(filename, 'r') as f:
    obj = json.loads(f.read())
  
    return obj

#------ Variables -------------------------------------------------------------+
active = ['aapl', 'amd', 'amzn', 'atvi', 'ea', 'msft', 'nvda', 'roku', 'lub', 'fb']
for stock in active:
    batch_size = 5
    days       = 50
    fit        = True
    epochs     = 1000
    #stock      = 'amzn'
    req        = True #make new api request?
    
    stock_path = f'./{stock}/'
    
    #------ Get Stock Prices ------------------------------------------------------+
    if req:
      symbol = stock.upper()
      # api-endpoint 
      URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=T7XYX1RHMEOB36OS'
      
      # sending get request and saving the response as response object 
      r = requests.get(url = URL)
        
      # extracting data in json format 
      data = r.json()
    
      save_json(data["Time Series (Daily)"], stock_path+f'{stock}.json')
    
    df = pd.read_json(stock_path+f'{stock}.json')
    y = []
    i = 0
    for date in df:
      if i < days:
        print(date, df[date]['4. close'])
        y.append(df[date]['4. close'])
      i+=1
    x = [i for i in range(len(y))]    #day
    y = y[::-1]                       #closing price
    true_y = y
    plt.plot(x, y, label=f'{stock}')
    
    #------ Create Model ----------------------------------------------------------+
        
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
    
    #------ Preprocess ------------------------------------------------------------+
    mean_adj = min(y)
    y = [val-mean_adj for val in y]
    x_train = np.array([np.array([i]) for i in x])
    y_train = np.array([np.array([j]) for j in y])
    x_train = np.reshape(x_train, (len(x),1,1))
    
    #------ Train/Fit Model -------------------------------------------------------+
    if fit:
      ckpt = ModelCheckpoint(stock_path+'best_model.h5', monitor='loss', \
        verbose=0, save_weights_only=True, save_best_only=True)
    
      model.fit(x_train, y_train, verbose=1, batch_size=batch_size, epochs=epochs, callbacks=[ckpt])
      model.save_weights(stock_path+f'{stock}.h5')
    model.load_weights(stock_path+'best_model.h5')
    
    #------ Plot NN Predictions ---------------------------------------------------+
    x = []
    y = []
    for i in range(len(true_y)):
      x.append(i)
      guess = model.predict(np.array(x).reshape(i+1, 1, 1))[-1][0]
      y.append(guess + mean_adj)
    
    plt.plot(x, y, label='prediction')
    plt.legend(loc=3)
    
    #------ Today vs Tomorrow Prediction ------------------------------------------+
    day = len(x)
    guesses = model.predict(np.array([i for i in range(day+1)]).reshape(day+1, 1, 1))
    today = guesses[-2][0]
    tomorrow = guesses[-1][0]
    print('Todays close:', true_y[-1], 'Guess:', today+mean_adj)
    print('Tomorrows guess:', tomorrow+mean_adj)
    
    #------ Show plots and correlation --------------------------------------------+
    r2 = metrics.r2_score(true_y, y)
    r =  pow(r2, .5)
    print('R:', r)
    now = datetime.datetime.now()
    month, day = now.month, now.day
    plt.savefig(stock_path+f'{month}-{day}')
    plt.show()
    
    with open(stock_path+f'pred_{month}-{day}', 'w') as file:
      out = f'{stock}, R:{r}' + '\n'
      out += f'{month}-{day} Close: ' + str(true_y[-1]) + '\n'
      out += f"Next Day Guess: {tomorrow+mean_adj}"
      file.write(out)