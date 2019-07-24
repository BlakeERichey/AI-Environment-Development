import gym
import rename

env = gym.make('rename-v0')

print('Observation Space:', env.observation_space)
print('Available Actions:', env.action_space.n   )
for epoch in range(1):
  num_steps = 0
  done      = False
  envstate  = env.reset()
  while not done and num_steps < 10: #perform action/step
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    num_steps += 1

env.close()