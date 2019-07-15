import gym
from   gym import spaces

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

def merge_tuple(arr): #arr: (('aa', 'bb'), 'cc') -> ('aa', 'bb', 'cc')
  return tuple(j for i in arr for j in (i if isinstance(i, tuple) else (i,)))

class Images(gym.Env):

  def __init__(self):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    shape = X_train[0].shape #(28, 28)

    self.action_space      = spaces.Discrete(10) #0-9 numbers
    self.observation_space = spaces.Box(low=0, high=255, \
      shape=(merge_tuple( (shape, 1) ))\
      ,dtype=np.float32)

    self.data = X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    self.answer = y_train 
    self.done = False
    self.steps = 0

  def step(self, action):
    reward = self.get_reward(action)

    self.steps += 1 
    return self.data[self.steps-1], reward, True, {}

    

  def reset(self,):
    return self.data[self.steps]   
  
  def render(self,):
    pass
  
  def get_reward(self, action):
    if action == np.argmax(self.answer[self.steps]):
      reward = 1
    else:
      reward = -1
    
    return reward