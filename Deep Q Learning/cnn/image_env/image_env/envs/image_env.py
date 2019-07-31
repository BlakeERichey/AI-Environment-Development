import gym, random
from   gym import spaces

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

def merge_tuple(arr): #arr: (('aa', 'bb'), 'cc') -> ('aa', 'bb', 'cc')
  return tuple(j for i in arr for j in (i if isinstance(i, tuple) else (i,)))

class Images(gym.Env):

  def __init__(self):
    print('Initializing Environment. Aquiring Mnist Data...')
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
    
    self.validation = X_test
    self.validation_answers = y_test

    self.answer = y_train 
    self.done = False
    self.steps = 0
    self.max_steps = 59999

  def step(self, action):
    # print('guess:', action, 'answer:', np.argmax(self.answer[self.steps]), action==np.argmax(self.answer[self.steps]))
    self.action = action
    if self.steps % 100 == 0:
      print(self.steps)
    reward = self.get_reward(action)

    self.steps += 1
    self.done = True
    return self.data[self.index], reward, self.done, {}

    

  def reset(self,):
    self.done = False
    self.reward = None
    self.index = random.randrange(0, 60000)
    return self.data[self.index]
  
  def render(self,):
    print('Guess:', self.action, 'Actual', np.argmax(self.answer[self.index]), 'Reward', self.reward)
  
  def get_reward(self, action):
    if not self.done:
        
      if action == np.argmax(self.answer[self.index]):
        reward = 1
      else:
        reward = 0
      
      self.reward = reward
    
      return reward