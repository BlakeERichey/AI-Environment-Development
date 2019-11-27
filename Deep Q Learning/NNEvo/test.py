import gym, traceback
from nn_evo_transfer import NNEvo
from time import time

config = {
  'tour': 4,
  'cores': 1,
  'cxrt': .2,
  'layers': 3, 
  'env': 'MountainCar-v0', 
  'elitist': 3,
  'sharpness': 3,
  'cxtype': 'splice',
  'population': 42, 
  'mxrt': 0.0001,
  'transfer': False,
  'generations': 30, 
  'mx_type': 'default',
  'selection': 'tour',
  'fitness_goal': -110,
  'random_children': 2,
  'validation_size': 10,
  'activation': 'softmax', 
  'nodes_per_layer': [256,512,256], 
}

#test model
try:
    agents = NNEvo(**config)
    agents.evaluate('MountainCar.h5')
except:
    traceback.print_exc()
    agents.envs[0].close()