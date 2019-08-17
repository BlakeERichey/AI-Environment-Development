
import gym
from NNEvo import NNEvo

env = gym.make('CartPole-v0')
print('Environment created')
config = {
  'tour': 4, 
  'cxrt': .1,
  'mxrt': 1,
  'layers': 1, 
  'env': env, 
  'elitist': 2,
  'cxtype': 'splice',
  'population': 5, 
  'generations': 100, 
  'selection': 'tour',
  'fitness_goal': 200,
  'validation_size': 8,
  'activation': 'softmax', 
  'nodes_per_layer': [10], 
}

#@profile
def train():
    agents = NNEvo(**config)
    agents.train()
    agents.show_plot()

def evaluate():
    agents = NNEvo(**config)
    agents.evaluate('best_model.h5')
    
train()
evaluate()