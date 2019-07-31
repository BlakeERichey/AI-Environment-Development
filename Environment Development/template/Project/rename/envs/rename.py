import gym
import numpy as np
from   gym   import error, spaces

class Rename(gym.Env):

  def __init__(self):
    '''
      Initialize environment variables
    '''

    self.action_space      = spaces.Discrete(10)
    self.observation_space = spaces.Box(low=-100, high=100, \
      shape=(10,),dtype=np.float32)

  def step(self, action):
    '''
      Offers an interface to the environment by performing an action and 
      modifying the environment via the specific action taken

      Returns: envstate, reward, done, info
    '''
    pass

  def reset(self,):
    ' Returns environment state after reseting environment variables '
    pass
  
  def render(self,):
    '''
      This method will provide users with visual representation of what is
      occuring inside the environment
    '''
    pass
  
  def get_reward(self):
    '''
      Calculate and return reward based on current environment state
    '''
    pass