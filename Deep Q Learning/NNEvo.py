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
from BRprofiler import profile


class NNEvo:

  def __init__(self, 
    tour=3, 
    mxrt=.01, 
    layers=1, 
    env=None, 
    elitist=3,
    population=10, 
    generations=10, 
    activation='linear', 
    nodes_per_layer=[4], 
    fitness_goal=200):

    '''
      config = {
        'tour': 3, 
        'mxrt': .01, 
        'layers': 1, 
        'env': None, 
        'elitist': 3,
        'population': 10, 
        'generations': 10, 
        'activation': 'linear', 
        'nodes_per_layer': [4], 
        'fitness_goal': 200
      }
    '''

    self.default_nodes   = 20
    self.env             = env
    self.mxrt            = mxrt
    self.best_fit        = None
    self.tour            = tour
    self.goal_met        = False
    self.num_layers      = layers
    self.elitist         = elitist
    self.activation      = activation 
    self.pop_size        = population
    self.generations     = generations
    self.fitness_goal    = fitness_goal
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
      model.summary()
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
      qvals = model.predict(envstate.reshape(1, -1))[0]
      action = np.argmax(qvals)
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
      if self.fitness_goal is not None and fitness >= self.fitness_goal:
        self.goal_met = self.models[i] #save model that met goal
        self.best_fit = (i, fitness)
        break

    if not self.goal_met:  #if goal met prepare to terminate
      ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)
      print('Ranked:', ranked)
      self.best_fit = ranked[0]

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
    for ind, individual in enumerate(population):
      for i, gene in enumerate(individual):
        if ind == len(population) - 1: #Randomly initialize last child
          mxrt = 1
        else:
          mxrt = self.mxrt
        if random.random() < self.mxrt:
          individual[i] = random.uniform(-1, 1)
    
    return population

  def train(self):
    self.create_population()
    print('Population created', len(self.pop))
    for i in range(self.generations):
      print('Generation:', i)
      parents = self.selection()
      if not self.goal_met:
        print('Goal not met. Parents selected.')
        print('Best fit:', self.best_fit)
        children = self.crossover(parents)
        print('Breeding done.')
        new_pop = self.mutate(children)
        print('Mutations done.')
        
        print('New pop:', len(new_pop))
        for i, individual in enumerate(new_pop):
          self.models[i].set_weights(self.deserialize(individual))
      else:
        print('Goal met!')
        self.goal_met.save_weights('best_model.h5')
        print('Best results saved to best_model.h5')
        break
    
    if not self.goal_met:
      if self.best_fit:
        self.models[self.best_fit[0]].save_weights('best_model.h5')
        print('Best results saved to best_model.h5')


  def evaluate(self, filename=None):
    if self.goal_met or filename:
      #load model
      if filename:
        model = self.create_nn()
        model.load_weights(filename)
      else:
        model = self.goal_met

      #display results
      while True:
        done = False
        rewards = []
        envstate = self.env.reset()
        while not done:
          qvals = model.predict(envstate.reshape(1, -1))[0]
          action = np.argmax(qvals)
          envstate, reward, done, info = self.env.step(action)
          self.env.render()
          rewards.append(reward)

        print('Reward:', sum(rewards))
        rewards = []


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

@profile
def train():
    env = gym.make('MountainCar-v0')
    print('Environment created')
    
    config = {
      'tour': 4, 
      'mxrt': .01, 
      'layers': 3, 
      'env': env, 
      'elitist': 8,
      'population': 50, 
      'generations': 100, 
      'activation': 'linear', 
      'nodes_per_layer': [64,64,64], 
      'fitness_goal': -100
    }
    
    agents = NNEvo(**config)
    agents.train()

def evaluate():
    env = gym.make('CartPole-v0')
    print('Environment created')
    agents = NNEvo(env=env)
    agents.evaluate('best_model.h5')
    
train()
#evaluate()