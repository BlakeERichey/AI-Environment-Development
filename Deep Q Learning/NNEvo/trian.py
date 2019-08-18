
import gym
from NNEvo import NNEvo
from time import time

env = gym.make('MountainCar-v0')
print('Environment created')
config = {
  'tour': 6, 
  'cxrt': .1,
  'mxrt': 1,
  'layers': 2, 
  'env': env, 
  'elitist': 3,
  'cxtype': 'splice',
  'population': 20, 
  'generations': 200, 
  'selection': 'tour',
  'fitness_goal': -110,
  'validation_size': 8,
  'activation': 'linear', 
  'nodes_per_layer': [256,256,256], 
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