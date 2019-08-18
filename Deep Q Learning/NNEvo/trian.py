
import gym
from NNEvo import NNEvo
from time import time

env = gym.make('MountainCar-v0')
print('Environment created')
config = {
  'tour': 8, 
  'cxrt': .1,
  'mxrt': 1,
  'layers': 3, 
  'env': env, 
  'elitist': 3,
  'cxtype': 'splice',
  'population': 15, 
  'generations': 200, 
  'selection': 'tour',
  'fitness_goal': -110,
  'validation_size': 8,
  'activation': 'softmax', 
  'nodes_per_layer': [256,256,128], 
}

#@profile
def train():
    agents = NNEvo(**config)
    agents.train()
    agents.show_plot()

def evaluate():
    agents = NNEvo(**config)
    agents.evaluate('best_model.h5')

start = time()
train()
end = time()
print('Time training:', end-start)
evaluate()