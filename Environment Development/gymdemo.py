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
    ps.print_stats(.05)
    print(s.getvalue())
    return retval

  return inner

import gym, time
#import gym_pull

@profile
def main():
#  gym_pull.pull('github.com/ppaquette/gym-doom')        # Only required once, envs will be loaded with import gym_pull afterwards
#  env = gym.make('ppaquette/DoomBasic-v0')
  env = gym.make('DoomCorridor-v0')
  print(env.action_space)
  print(env.observation_space)
  for _ in range(20):  #game number
    env.reset()
    for t in range(9):   #turn counter
      action = env.action_space.sample() 
      _, _, done, _ = env.step(action) #observation, reward, done, info
      env.render()
      time.sleep(1)
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