from gym.envs.registration import register

try:
  register(
      id='rename-v0',
      entry_point='rename.envs:Rename',
  )
except:
  pass