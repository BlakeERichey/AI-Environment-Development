import gym
import rename

env = gym.make('rename-v0')

action = env.action_space.sample()
print(action)