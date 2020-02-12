import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, TimeDistributed
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

env = gym.make('Breakout-v0')
print(env.unwrapped.get_action_meanings())

class ActorCritic:
  
  def __init__(self,env, n_timesteps):
    model = Sequential()
    #========== Convolutional Base ==============================
    #input layer
    model.add(TimeDistributed( Conv2D(64, kernel_size=3, activation='relu'), \
      input_shape=(n_timesteps,) + env.observation_space.shape ))

    model.add(TimeDistributed( Conv2D(32, kernel_size=3, activation='relu') ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    model.add(TimeDistributed( Conv2D(16, kernel_size=3, activation='relu') ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    model.add(TimeDistributed(Flatten()))

    #========== FCN ==============================
    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(LSTM(16, return_sequences=True, dropout=.2))
    model.add(LSTM(8, return_sequences=True))
    model.add(TimeDistributed(Dense(env.action_space.n, activation='softmax')))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mse'])

    model.summary()
    self.model = model

def run_game(steps=None, n_timesteps = 1):
  ob = env.reset()
  ob = np.array([ob for _ in range(n_timesteps)])
  print('Reset.')
  done = False
  num_steps = 0
  while not done and (True,num_steps<steps)[steps is not None]:
    qvals = model.predict(np.expand_dims(ob, axis=0))
    action = np.argmax(qvals[0][-1]) #action based on last time steps decision
    print(action)
    action = env.action_space.sample()
    envstate, reward, done, _ = env.step(action) 

    #add to timestep oberservation
    ob = np.concatenate((ob, np.expand_dims(envstate, axis=0)), axis=0)[1:]
    num_steps+=1
    env.render()
  env.close()


num_steps = 200
n_timesteps = 4
model = ActorCritic(env, n_timesteps).model

run_game(num_steps, n_timesteps)