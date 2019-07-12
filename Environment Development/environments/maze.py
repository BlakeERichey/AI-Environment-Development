import cProfile, pstats, io
def profile(fnc):
  """A decorator that uses cProfile to profile a function"""
  
  def inner(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    retval = fnc(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(.15)
    print(s.getvalue())
    return retval

  return inner

import gym
import time
import gym_maze
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU


def build_model(maze, lr=0.001, num_actions=4):
  model = Sequential()
  model.add(Dense(maze.size, input_shape=(maze.size,)))
  model.add(PReLU())
  model.add(Dense(maze.size))
  model.add(PReLU())
  model.add(Dense(num_actions))
  model.compile(optimizer='adam', loss='mse')
  return model


@profile
def main(maze):
  env = gym.make('maze-v0')
  env.__init__(maze)
  print(env.action_space)
  # print(env.observation_space)
  model = build_model(maze, num_actions=env.action_space.n)
  model.load_weights('model.h5')
  for _ in range(1):  #game number
    observation = env.reset()
    total_reward = 0
    for t in range(200):   #turn counter
      # action = env.action_space.sample() 
      action = np.argmax(model.predict(observation)[0])
      print('move number', t+1, 'action:', action)
      observation, reward, done, _ = env.step(action) #observation, reward, done, info
      
      total_reward+=reward
      env.render()
      time.sleep(.5)
      if done in ['win', 'lose']:
        break
        # print(f'Game finished after {t+1} turns\n')
    print('Reward:', total_reward)
    print(done if done in ['win', 'lose'] else 'Lost by timeout.')
    env.close()

maze =  np.array([
    [ 1.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  1.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

main(maze)