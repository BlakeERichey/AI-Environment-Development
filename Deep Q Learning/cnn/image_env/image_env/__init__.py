from gym.envs.registration import register

try:
  register(
      id='image_env-v0',
      entry_point='image_env.envs:Images',
  )

  register(
      id='image_env-v1',
      entry_point='image_env.envs:Images2',
  )
except:
  pass