from model import ActorCritic
from train import train, make_env
import gym
import numpy as np 

env_params = {
  'env_name': 'Breakout-v0',
  'num_steps': 200,
  'seed': 1,
}

env = make_env(env_params)

#create critic
n_timesteps = 4 #how many frames agent gets
critic = ActorCritic(env, n_timesteps).create_critic()

train(0, env_params, critic, n_timesteps=n_timesteps)