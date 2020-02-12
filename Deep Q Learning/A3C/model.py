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
  '''
    Create actor-critic models for A3C algorithm
  '''
  def __init__(self,env, n_timesteps):
    #Actor
    actor = Sequential()
    #========== Convolutional Base ==============================
    #input layer
    actor.add(TimeDistributed( Conv2D(64, kernel_size=3, activation='relu'), \
      input_shape=(n_timesteps,) + env.observation_space.shape ))

    actor.add(TimeDistributed( Conv2D(32, kernel_size=3, activation='relu') ))
    actor.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    actor.add(TimeDistributed( Conv2D(16, kernel_size=3, activation='relu') ))
    actor.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    actor.add(TimeDistributed(Flatten()))

    #========== FCN ==============================
    actor.add(LSTM(32, return_sequences=True))
    actor.add(TimeDistributed(BatchNormalization()))
    actor.add(LSTM(16, return_sequences=True, dropout=.2))
    actor.add(LSTM(8, return_sequences=True))
    actor.add(TimeDistributed(Dense(env.action_space.n, activation='softmax')))
    actor.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mse'])

    actor.summary()
    self.actor = actor
    
    #Critic
    critic = Sequential()
    #========== Convolutional Base ==============================
    #input layer
    critic.add(TimeDistributed( Conv2D(64, kernel_size=3, activation='relu'), \
      input_shape=(n_timesteps,) + env.observation_space.shape ))

    critic.add(TimeDistributed( Conv2D(32, kernel_size=3, activation='relu') ))
    critic.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    critic.add(TimeDistributed( Conv2D(16, kernel_size=3, activation='relu') ))
    critic.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
    critic.add(TimeDistributed(Flatten()))

    #========== FCN ==============================
    critic.add(LSTM(32, return_sequences=True))
    critic.add(TimeDistributed(BatchNormalization()))
    critic.add(LSTM(16, return_sequences=True, dropout=.2))
    critic.add(TimeDistributed(Dense(1, activation='linear')))
    critic.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mse'])

    critic.summary()
    self.critic = critic

  def feed_forward(self, ob):
    '''
      Used for easy computation of both actor and critics evalution of a state
    '''
    #actor determines action
    qvals = self.actor.predict(np.expand_dims(ob, axis=0))
    action = np.argmax(qvals[0][-1]) #action based on last time steps decision

    #critic evaluates state value
    state_value = self.critic.predict(np.expand_dims(ob, axis=0))[0][-1]

    return action, state_value


def run_game(steps=None, n_timesteps = 1):
  '''
    Runs games using a ActorCritic class obj as `agents` on a gym environment
  '''

  ob = env.reset()
  ob = np.array([ob for _ in range(n_timesteps)])
  print('Reset.')
  done = False
  num_steps = 0
  while not done and (True,num_steps<steps)[steps is not None]:
    action, state_value = agents.feed_forward(ob)
    print(action, state_value)
    action = env.action_space.sample()
    envstate, reward, done, _ = env.step(action) 

    #add to timestep oberservation
    ob = np.concatenate((ob, np.expand_dims(envstate, axis=0)), axis=0)[1:]
    num_steps+=1
    env.render()
  env.close()


# num_steps = 200
# n_timesteps = 4
# agents = ActorCritic(env, n_timesteps) 

# run_game(num_steps, n_timesteps)