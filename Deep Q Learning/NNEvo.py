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

  def __init__(self, 
    tour=3, 
    cxrt=.2,
    mxrt=.01, 
    layers=1, 
    env=None, 
    elitist=3,
    cxtype='avg',
    population=10, 
    generations=10, 
    selection='tour',
    model_type='ann',
    fitness_goal=200,
    validation_size=0,
    activation='linear', 
    nodes_per_layer=[4]):

    '''
      config = {
        'tour': 3, 
        'cxrt': .2,
        'mxrt': .01,
        'layers': 1, 
        'env': None, 
        'elitist': 3,
        'cxtype': 'avg',
        'population': 10, 
        'generations': 10,
        'selection': 'tour',
        'model_type': 'ann', 
        'fitness_goal': 200,
        'validation_size': 0,
        'activation': 'linear', 
        'nodes_per_layer': [4], 
      }
    '''

    self.default_nodes   = 20
    self.env             = env
    self.mxrt            = mxrt
    self.cxrt            = cxrt   #chance of parent being selected (crossover rate)
    self.best_fit        = None
    self.tour            = tour
    self.cxtype          = cxtype
    self.goal_met        = False
    self.num_layers      = layers
    self.elitist         = elitist
    self.selection_type  = selection  #selection type (cxrt/tour)
    self.activation      = activation 
    self.model_type      = model_type
    self.pop_size        = population
    self.generations     = generations
    self.fitness_goal    = fitness_goal
    self.validation_size = validation_size
    self.nodes_per_layer = nodes_per_layer
    self.num_outputs     = self.env.action_space.n
    self.num_features    = self.env.observation_space.shape[0]

    self.models = [] #list of individuals
    self.pop    = [] #population (2d-list of weights)
    self.weight_shapes   = None
    self.weights_lengths = None
    self.plots = [] #points for matplotlib
    self.episodes = 0

    
    if self.model_type == 'cnn':
      self.envshape       = self.env.observation_space.shape
      self.batch_envshape = merge_tuple((1, self.envshape))
      self.pool_size       = 2
      self.filter_size     = 3
      self.stride_size     = None 

  #--- Initialize Population --------------------------------------------------+
  def create_nn(self):
    '''Create individual of population'''

    if self.model_type == 'ann':

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
    
    elif self.model_type == 'cnn':
      model = Sequential()

      for layer in range(self.num_layers):

        try:
          nodes=self.nodes_per_layer[layer]
        except IndexError:
          nodes = None

        if nodes is None:
          nodes = self.default_nodes

        if layer == 0:
          #input layer
          model.add(Conv2D(nodes, kernel_size=self.filter_size, activation='relu', \
            input_shape=(self.envshape)))
        else:
          #add hidden layers
          model.add(Conv2D(nodes, kernel_size=self.filter_size, activation='relu'))

      model.add(MaxPooling2D(pool_size=self.pool_size, strides=self.stride_size))
      model.add(Flatten())
      #output layer
      model.add(Dense(self.num_outputs, activation='softmax'))

      #compile model using accuracy to measure model performance
      model.compile(optimizer=Adam(lr=0.001), \
        loss='categorical_crossentropy', metrics=['accuracy'])


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
      if self.mxrt is 1:
        self.mxrt = 1/( self.weights_lengths[-1] * 2 )
      print('Weight Lengths:', self.weights_lengths)
      print('Mutation Rate:', self.mxrt)
      print('Crossover Type:', self.cxtype)
      print('Selection Type:', self.selection_type)
    return model
  
  def create_population(self):
    for _ in range(self.pop_size):
      model = self.create_nn()
      self.models.append(model)
      self.pop.append(self.serialize(model))
  #----------------------------------------------------------------------------+

  #--- Fitness Calculation ----------------------------------------------------+

  def quality(self, model):
    '''
      fitness function. Returns quality of model
      Runs 1 episode of environment
    '''

    self.episodes += 1
    done = False
    rewards = []
    envstate = self.env.reset()
    while not done:
      qvals = model.predict(self.adj_envstate(envstate))[0]
      action = np.argmax(qvals)
      envstate, reward, done, info = self.env.step(action)
      rewards.append(reward)
    
    return sum(rewards)
  
  #----------------------------------------------------------------------------+
  
  #--- Breed Population -------------------------------------------------------+
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
        if self.validation_size:
          valid = self.validate(self.models[i])
        else:
          valid = True
        
        if valid:
          self.goal_met = self.models[i] #save model that met goal
          self.best_fit = (i, fitness)
          break

    if not self.goal_met:  #if goal met prepare to terminate
      ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)
      print('Ranked:', ranked)
      self.best_fit = ranked[0]

      for i in range(self.elitist):
        selection.append(ranked[i])

      if self.selection_type == 'tour':
        while len(selection) < self.pop_size:
          tourny = random.sample(ranked, self.tour)
          selection.append(max(tourny, key=lambda x:x[1]))

      elif self.selection_type == 'cxrt':
        while len(selection) < self.pop_size:
          for model in ranked:
            if random.random() < self.cxrt:
              selection.append(model)
            

    self.plots.append(self.best_fit)
    return selection

  def crossover(self, parents):
    children = [] #gene strings

    #keep elites
    for i in range(self.elitist):
      index = parents[i][0]
      children.append(self.pop[index])

    parents = random.sample(parents, len(parents)) #randomize breeding pool

    #breed rest
    i = 0
    while len(children) < self.pop_size:
      parent1 = parents[i]
      parent2 = parents[len(parents)-i-1]

      parent1_genes = self.pop[parent1[0]]
      parent2_genes = self.pop[parent2[0]]
      if self.cxtype == 'splice':
        #splice genes
        geneA = int(random.random() * len(parent1_genes))
        geneB = int(random.random() * len(parent1_genes))

        child = splice_list(parent1_genes, parent2_genes, geneA, geneB)
      else:
        child = ((np.array(parent1_genes) + np.array(parent2_genes)) / 2).tolist()
      
      children.append(child)
      i+=1
    
    return children
  
  def mutate(self, population):
    for ind, individual in enumerate(population):
      for i, gene in enumerate(individual):
        mxrt = self.mxrt
        if self.pop_size > 10:
          if ind == len(population) - 1: #Randomly initialize last child
            mxrt = 1
        if random.random() < mxrt:
          individual[i] = random.uniform(-1, 1)
    
    return population
  #----------------------------------------------------------------------------+
  
  #--- Train/Evaluate ---------------------------------------------------------+

  def train(self):
    self.create_population()
    print('Population created', len(self.pop))
    for i in range(self.generations):
      print('\nGeneration:', i)
      parents = self.selection()
      if not self.goal_met:
        print('Goal not met. Parents selected.')
        print('Best fit:', self.best_fit)
        children = self.crossover(parents)
        print('Breeding done.')
        new_pop = self.mutate(children)
        print('Mutations done.')
        
        print('New pop:', len(new_pop))
        self.pop = new_pop
        for i, individual in enumerate(new_pop):
          self.models[i].set_weights(self.deserialize(individual))
      else:
        print(f'Goal met! Episodes: {self.episodes}')
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
        print(f'Weights loaded from {filename}')
      else:
        model = self.goal_met

      #display results
      while True:
        done = False
        rewards = []
        envstate = self.env.reset()
        while not done:
          qvals = model.predict(self.adj_envstate(envstate))[0]
          action = np.argmax(qvals)
          envstate, reward, done, info = self.env.step(action)
          self.env.render()
          rewards.append(reward)

        print('Reward:', sum(rewards))
        rewards = []
  #----------------------------------------------------------------------------+

  #--- Validate Fitness -------------------------------------------------------+
  def validate(self, model):
    print('Validating Model...', end='')
    
    total_rewards = []
    n_epochs = self.validation_size
    #test results
    for epoch in range(n_epochs):
      done = False
      rewards = []
      envstate = self.env.reset()
      while not done:
        qvals = model.predict(self.adj_envstate(envstate))[0]
        action = np.argmax(qvals)
        envstate, reward, done, info = self.env.step(action)
        rewards.append(reward)

      total_rewards.append(sum(rewards))
    print(sum(total_rewards)/len(total_rewards))
    return sum(total_rewards)/len(total_rewards) >= self.fitness_goal
  #----------------------------------------------------------------------------+

  #--- Graph Functions --------------------------------------------------------+

  def show_plot(self):
    y = [self.plots[i][1] for i in range(len(self.plots))] #best fitness
    x = [i for i in range(len(self.plots))] #generation

    plt.plot(x, y, label='Best fitness')
    plt.legend(loc=4)
    plt.show()
    
  #----------------------------------------------------------------------------+

  #--- Helper Functions -------------------------------------------------------+

  def adj_envstate(self, envstate):
    '''
      changes envstate to a predict acceptable format
    '''

    if self.model_type == 'cnn':
      adj_envstate = envstate.reshape(self.batch_envshape)
    else:
      adj_envstate = envstate.reshape(1, -1)
    
    return adj_envstate

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

def merge_tuple(arr): #arr: (('aa', 'bb'), 'cc') -> ('aa', 'bb', 'cc')
  return tuple(j for i in arr for j in (i if isinstance(i, tuple) else (i,)))
#------------------------------------------------------------------------------+



import gym
import image_env
from time import time

env = gym.make('image_env-v1')
print('Environment created')
config = {
  'tour': 4, 
  'cxrt': .1,
  'mxrt': 1,
  'layers': 3, 
  'env': env, 
  'elitist': 3,
  'cxtype': 'splice',
  'population': 20, 
  'generations': 150, 
  'selection': 'tour',
  'model_type': 'cnn',
  'fitness_goal': 240,
  'validation_size': 2,
  'activation': 'softmax', 
  'nodes_per_layer': [64,32,32], 
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
#evaluate()