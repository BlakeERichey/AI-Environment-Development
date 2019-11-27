import gym, traceback
from nn_evo_transfer import NNEvo
from time import time

config = {
  'tour': 3,
  'cores': 3,
  'cxrt': .2,
  'layers': 0, 
  'env': 'BattleZone-v0', 
  'elitist': 3,
  'sharpness': 1,
  'cxtype': 'splice',
  'population': 21, 
  'mxrt': 0.00001,
  'transfer': True,
  'generations': 80, 
  'mx_type': 'default',
  'selection': 'tour',
  'fitness_goal': 6000,
  'random_children': 1,
  'validation_size': 2,
  'activation': 'linear', 
  'nodes_per_layer': [], 
}

#test model
try:
    agents = NNEvo(**config)
    agents.evaluate('BattleZoneTemp.h5')
except:
    traceback.print_exc()
    agents.envs[0].close()