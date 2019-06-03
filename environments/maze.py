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
    ps.print_stats()
    print(s.getvalue())
    return retval

  return inner

import gym
import time
import gym_maze
import numpy as np

@profile
def main(maze):
  env = gym.make('maze-v0')
  env.__init__(maze)
  print(env.action_space)
  print(env.observation_space)
  for _ in range(1):  #game number
    env.reset()
    for t in range(200):   #turn counter
      action = env.action_space.sample() 
      print('move number', t+1, 'action:', action)
      _, _, done, _ = env.step(action) #observation, reward, done, info
      if env.render():
        pass
      time.sleep(.5)
      if done in ['win', 'lose']:
        break
        # print(f'Game finished after {t+1} turns\n')
    print(done if done in ['win', 'lose'] else 'Lost by timeout.')
    env.close()

maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

main(maze)