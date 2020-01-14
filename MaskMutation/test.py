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

elites = 5
pop_size = 30
generations = 200

ga = Evolution(generations, pop_size, elites, goal=-110, metric='valid')
ga.create_species(net, mutations=.1, patience=10)

worker = ga.train(env, validate=True, render=False, return_worker=True)


reward, _ = worker.fitness(env,episodes=100,render=True)
print("Average reward:", reward)

env.close()

# for i, worker in enumerate(ga.workers):
#   print('Worker', i, worker.history)