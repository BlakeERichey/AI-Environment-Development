'''
Created on Friday August 16, 2019

@author: Blake Richey

Implementation of NeuroEvolution Algorithm:
  Develop a neurel network and implement genetic algorithm that finds optimum
  weights, as an alternative to backpropogation
'''

import gym, operator
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
from   tensorflow.keras.optimizers import Adam
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.python.client    import device_lib
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM


class NNEvo:

  def __init__(self, population=100, mxrt=.01, crrt=.2, layers=3, nodes_per_layer=[10,10,11], generations=2, env=None, activation='linear', tour=3, elitist=3):
    self.default_nodes   = 20
    self.env             = env
    self.crrt            = crrt
    self.mxrt            = mxrt
    self.tour            = tour
    self.num_layers      = layers
    self.elitist         = elitist
    self.activation      = activation 
    self.pop_size        = population
    self.generations     = generations
    self.nodes_per_layer = nodes_per_layer
    self.num_outputs     = self.env.action_space.n
    self.num_features    = self.env.observation_space.shape[0]

    self.models = [] #list of individuals
    self.pop    = [] #population (2d-list of weights)
    self.weight_shapes   = None
    self.weights_lengths = None

  #--- Initialize Population --------------------------------------------------+
  def create_nn(self):
    '''Create individual of population'''

    model = Sequential()
    model.add(Dense(self.num_features, input_shape = (self.num_features,)))
    
    for layer in range(self.num_layers):

        try:
            nodes=self.nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = self.default_nodes

        model.add(Dense(units = nodes, activation = 'relu'))
    
    #output layer
    model.add(Dense(units = self.num_outputs, activation = self.activation, name='dense_output'))
    model.compile(optimizer = Adam(lr=0.001), loss = 'mse', metrics=['accuracy'])

    #create deserialize dependencies
    if self.weight_shapes is None:
      self.weight_shapes = []
      self.weights_lengths = []

      weights = model.get_weights()
      for x in weights:
        self.weight_shapes.append(x.shape)

        #generate indicies of weights to recreate weight structure from gene string
        length = len(x.reshape(1, -1)[0].tolist())
        if not self.weights_lengths:
          self.weights_lengths.append(length)
        else:
          self.weights_lengths.append(self.weights_lengths[len(self.weights_lengths)-1]+length)
    
    return model
  
  def create_population(self):
    for _ in range(self.pop_size):
      model = self.create_nn()
      self.models.append(model)
      self.pop.append(self.serialize(model))
  #----------------------------------------------------------------------------+

  def quality(self, model):
    '''
      fitness function. Returns quality of model

      Runs 1 episode of environment
    '''

    done = False
    rewards = []
    envstate = self.env.reset()
    while not done:
      action = model.predict(envstate.reshape(1, -1))
      envstate, reward, done, info = self.env.step(action)
      rewards.append(reward)
    
    return sum(rewards)
  
  def selection(self):
    '''
      generate mating pool, tournament && elistist selection policy
    '''
    selection = []

    ranked = [] #ranked models, best to worst
    for i, model in enumerate(self.models):
      fitness = self.quality(model)
      ranked.append((i, fitness))
    
    ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)

    for i in range(self.elitist):
      selection.append(ranked.pop(0))

    while len(selection) < self.pop_size:
      tourny = random.sample(ranked, self.tour)
      selection.append(max(tourny, key=lambda x:x[1]))
    
    return selection

  def crossover(self, parents):
    children = [] #gene strings

    #keep elites
    for i in range(self.elitist):
      index = parents[i][0]
      children.append(self.serialize(self.models[index]))

    #breed rest
    i = 0
    while len(children) < self.pop_size:
      parent1 = parents[i]
      parent2 = parents[len(parents)-i-1]

      parent1_genes = self.serialize(self.models[parent1[0]])
      parent2_genes = self.serialize(self.models[parent2[0]])

      #splice genes
      geneA = int(random.random() * len(parent1_genes))
      geneB = int(random.random() * len(parent1_genes))

      child = splice_list(parent1_genes, parent2_genes, geneA, geneB)
      children.append(child)
    
    return children
  
  def mutate(self, population):
    for individual in population:
      for i, gene in enumerate(individual):
        if random.random() > self.mxrt:
          individual[i] = random.uniform(-1, 1)


  def serialize(self, model):
    '''
      serializes model's weights into a gene string
    '''
    weights = model.get_weights()
    flattened = []
    for arr in weights:
      flattened+=arr.reshape(1, -1)[0].tolist()
    
    return flattened

  def deserialize(self, genes):
    '''
      deserializes gene string into weights
    '''
    shapes = self.weight_shapes
    lengths = self.weights_lengths
    
    weights = []
    for i, val in enumerate(lengths):
      if i == 0:
        begin = 0
      else:
        begin = lengths[i-1]
      weights.append(np.array(genes[begin:val]).reshape(shapes[i]))
    
    return weights

def splice_list(list1, list2, index1, index2):
  '''
    combined list1 and list2 taking splice from list1 with starting index `index1`
    and ending index `index2`
  '''
  if index1 == 0:
    splice = list1[index1:index2]
    splice += list2[index2:len(list1)]
  else:
    splice = list2[:index1] + list1[index1:index2]
    splice += list2[index2:len(list1)]
  
  return splice

# create individual(NN)/population(multiple NNs) -
# #mating pool/parent selection                  - 
# crossover/breed                                - 
# mutation                                       - 
# forward propgation-
# training
# activation-
#tour size and elit size from init               -
#solved, num episodes vars, best modal etc

# matingpool/parent selection/elistist/reoulette wheel/tournament
# crossover
# mutation
# create_individual
# create population
# determine best action/apply weights
# training


# env = gym.make('CartPole-v0')
# agents = NNEvo(env=env)
# model = agents.create_nn()
# weights = agents.flatten_weights(model)
# #print(weights)
# #print(len(weights))
# #print(agents.weights_lengths)

# weights = agents.weights_from_genes(weights)
# agents.model.set_weights(weights)
# print('success')