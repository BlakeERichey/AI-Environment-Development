from neural import Network
from worker import Worker
from evolution import Evolution
import numpy as np
import gym

env = gym.make("MountainCar-v0")
inputs = env.observation_space.shape[0]
outputs = env.action_space.n
print('Inputs:', inputs, "Action Space", outputs)

net = Network()
net.add_layer(inputs, "input", use_bias=False)
net.add_layer(256, "relu", use_bias=False)
net.add_layer(256, "relu", use_bias=False)
net.add_layer(256, "relu", use_bias=False)
net.add_layer(outputs, "softmax", use_bias=False)
net.compile()

goal = -110
elites = 5
pop_size = 30
sharpness = 1
generations = 200
metric='valid'
params = [generations, pop_size, elites, sharpness, goal, metric]
ga = Evolution(*params)
ga.create_species(net, mutations=.5, patience=10, alpha=.1)
worker = ga.train(env, validate=True, render=False, return_worker=True)
worker.fitness(env,episodes=25,render=True)

env.close()

for i, worker in enumerate(ga.workers):
  print('Worker', i, worker.history)