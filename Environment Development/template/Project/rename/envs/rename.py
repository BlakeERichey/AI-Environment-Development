import gym
import numpy as np
from   gym   import error, spaces

class Rename(gym.Env):

  def __init__(self):
    self.action_space      = spaces.Discrete(10)
    self.observation_space = spaces.Box(low=-100, high=100, \
      shape=(10,),dtype=np.float32)

  def step(self):
    pass

  def reset(self,):
    pass
  
  def render(self,):
    pass
  
  def get_reward(self):
    pass