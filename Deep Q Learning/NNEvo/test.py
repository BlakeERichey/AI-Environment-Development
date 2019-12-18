import gym, traceback
from nn_evo_transfer import NNEvo
from time import time

config = {
  'tour': 4,
  'cores': 1,
  'cxrt': .2,
  'layers': 0, 
  'env': 'BattleZone-v0', 
  'elitist': 4,
  'sharpness': 1,
  'cxtype': 'splice',
  'population': 40, 
  'mxrt': 'default',
  'transfer': True,
  'generations': 100, 
  'mx_type': 'default',
  'selection': 'tour',
  'fitness_goal': None,
  'random_children': 1,
  'validation_size': 0,
  'activation': 'softmax', 
  'nodes_per_layer': [], 
}

#test model
try:
    agents = NNEvo(**config)
    agents.evaluate('BattleZoneTemp.h5')
except:
    traceback.print_exc()
    agents.envs[0].close()