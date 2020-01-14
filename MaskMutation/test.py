from neural import Network
from worker import Worker
from evolution import Evolution
import numpy as np
import gym

env = gym.make("Cartpole-v0")

net = Network()
net.add_layer(4, "input", None, False)
net.add_layer(32, "linear", None, False)
net.add_layer(2, "linear", None, False)
net.compile()

goal = 100
elites = 2
pop_size = 5
sharpness = 1
generations = 2
params = [generations, pop_size, elites, sharpness, goal]
ga = Evolution(*params)
