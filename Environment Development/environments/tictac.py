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
import gym_tictac4

@profile
def main():
  env = gym.make('tictac4-v0')
  print(env.action_space)
  print(env.observation_space)
  for _ in range(100):  #game number
    env.reset()
    for t in range(9):   #turn counter
      action = env.action_space.sample() 
      _, _, done, _ = env.step(action) #observation, reward, done, info
      env.render()
      if done:
        break
        # print(f'Game finished after {t+1} turns\n')
    env.close()


main()

# from gym import spaces
# space = spaces.Discrete(8)
# x = space.sample()
# print(x)
# assert space.contains(x)
# assert space.n == 8

# from gym import envs
# print(envs.registry.all())