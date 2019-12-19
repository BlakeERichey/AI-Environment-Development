import gym, traceback
from nn_evo_transfer import NNEvo
from time import time

config = {
  'tour': 2,
  'cores': 12,
  'cxrt': .005,
  'layers': 2, 
  'env': 'MountainCar-v0', 
  'elitist': 3,
  'sharpness': 1,
  'cxtype': 'weave',
  'population': 30,
  'mxrt': 'default',
  'transfer': False,
  'generations': 200, 
  'mx_type': 'default',
  'selection': 'tour',
  'fitness_goal': -110,
  'random_children': 0,
  'validation_size': 100,
  'activation': 'linear', 
  'nodes_per_layer': [256,256],
}

#test model
try:
    agents = NNEvo(**config)
    agents.evaluate('mountaincar2_-119.h5')
except:
    traceback.print_exc()
    agents.envs[0].close()