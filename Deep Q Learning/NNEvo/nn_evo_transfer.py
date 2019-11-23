'''
Created on Wednesday November 11, 2019

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
from   time                          import time
from   tensorflow.keras.optimizers   import Adam
from   collections                   import deque
from   tensorflow.keras              import backend
from   sklearn.model_selection       import train_test_split
from   tensorflow.keras.applications import ResNet50
from   tensorflow.python.client      import device_lib
from   tensorflow.keras.models       import Sequential, Model
from   tensorflow.keras.callbacks    import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers       import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM


class NNEvo:

  def __init__(self, 
    tour=3, 
    cxrt=.2,
    layers=1, 
    env=None,
    elitist=3,
    sharpness=1, 
    cxtype='avg',
    population=10, 
    mxrt='default', 
    transfer=False,
    generations=10, 
    selection='tour',
    mx_type='default',
    random_children=1, 
    fitness_goal=200,
    validation_size=0,
    activation='linear',
    nodes_per_layer=[4]):

    '''
      config = {
        'tour': 3, 
        'cxrt': .2,
        'layers': 1, 
        'env': None, 
        'elitist': 3,
        'sharpness': 1,
        'cxtype': 'avg',
        'population': 10, 
        'mxrt': 'default',
        'transfer': False,
        'generations': 10, 
        'selection': 'tour',
        'fitness_goal': 200,
        'random_children': 1,
        'mx_type': 'default',
        'validation_size': 0,
        'activation': 'linear', 
        'nodes_per_layer': [4], 
      }
    '''

    self.default_nodes   = 20
    self.env             = env
    self.mxrt            = mxrt        #chance of a single weight being mutated
    self.cxrt            = cxrt        #chance of parent being selected (crossover rate)
    self.best_fit        = None        #(model, fitness) with best fitness
    self.tour            = tour        #tournament sample size when using tour selection policy
    self.cxtype          = cxtype      #cross over type (gene splicing or avging)
    self.goal_met        = False       #holds model number that meets fitness goal
    self.num_layers      = layers      #qty of hidden layers
    self.mx_type         = mx_type
    self.elitist         = elitist     #n best models transitioned into nxt gen
    self.transfer        = transfer    #implement transfer cnn
    self.sharpness       = sharpness   #epochs to run when evaluating fitness
    self.selection_type  = selection   #selection type (cxrt/tour)
    self.activation      = activation  #activation type for output layer
    self.pop_size        = population  #number of neural nets in population
    self.generations     = generations 
    self.fitness_goal    = fitness_goal #goal for fitness (episode score) to reach
    self.validation_size = validation_size #number of episodes to run to validate a models success in reaching a fitness goal
    self.nodes_per_layer = nodes_per_layer #list of qty of nodes in each hidden layer
    self.random_children = random_children #how many children to randomly mutate
    self.num_features    = self.env.observation_space.shape[0]

    
    outputs = 1
    if hasattr(env.action_space, 'n'):
      outputs = self.env.action_space.n
    self.num_outputs     = outputs

    self.models = [] #list of individuals 
    self.pop    = [] #population (2d-list of weights)
    self.weight_shapes   = None
    self.weights_lengths = None
    self.plots = [] #points for matplotlib
    self.episodes = 0
    

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
    model.add(Dense(units = self.num_outputs, activation = self.activation))
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
      if self.mxrt == 'default':
        self.mxrt = 1/( self.weights_lengths[-1] * 1.8 )
      print('Weight Lengths:', self.weights_lengths)
      print('Mutation Rate:', self.mxrt)
      print('Crossover Type:', self.cxtype)
      print('Selection Type:', self.selection_type)
      print('Sharpness:', self.sharpness)
    return model
  
  def create_transfer_cnn(self, ref_model=None, fcn_weights=None):
    '''creates resnet model. will load deserialized weights by passing in weights'''

    if not ref_model:
      self.env.observation_space.shape[0]
      model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.env.observation_space.shape))
      for layer in model.layers:
        layer.trainable = False
      
      pretrained_weights = model.get_weights()

      flattened = Flatten()(model.output)
      #Add FCN
      for layer in range(self.num_layers):

        try:
            nodes=self.nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = self.default_nodes

        if layer == 0:
          add_layer = Dense(units = nodes, activation = 'relu')(flattened)
        else:
          add_layer = Dense(units = nodes, activation = 'relu')(add_layer)
      
      if self.num_layers:
        output = Dense(units = self.num_outputs, activation = self.activation)(add_layer)
      else:
        output = Dense(units = self.num_outputs, activation = self.activation)(flattened)

      model = Model(model.inputs, output)
      model.compile(Adam(lr=1e-3), 'mse', metrics=['acc'])
    else:
      model = ref_model

      if fcn_weights:
        assert len(fcn_weights) == len(self.weight_shapes), \
          f'Invalid Weight Structure. Expected {len(self.weight_shapes)}, got {len(fcn_weights)}.'
        all_weights = model.get_weights()
        untrainable = all_weights[:-len(self.weight_shapes)]
        weights = all_weights[-len(self.weight_shapes):]
        # print('Deserialized weights length:', len(weights))
        for i, matrix in enumerate(weights):
          # print('Original', matrix)
          matrix[:] = fcn_weights[i]
          # print('Result', matrix)
      
        model.set_weights(untrainable + weights)
    
    #create deserialize dependencies
    if self.weight_shapes is None:
      model.summary()
      self.weight_shapes = []
      self.weights_lengths = []

      weights = model.get_weights()
      self.full_weights_length = len(weights)
      self.pretrained_weights_length = len(pretrained_weights)
      for i in range(len(pretrained_weights), len(weights)):
        self.weight_shapes.append(weights[i].shape)

        #generate indicies of weights to recreate weight structure from gene string
        length = len(weights[i].reshape(1, -1)[0].tolist())
        if not self.weights_lengths:
          self.weights_lengths.append(length)
        else:
          self.weights_lengths.append(self.weights_lengths[len(self.weights_lengths)-1]+length)
      if self.mxrt == 'default':
        self.mxrt = 1/( self.weights_lengths[-1] * 1.8 )
      print('Weight Shapes:', self.weight_shapes)
      print('Weight Lengths:', self.weights_lengths)
      print('Mutation Rate:', self.mxrt)
      print('Crossover Type:', self.cxtype)
      print('Selection Type:', self.selection_type)
      print('Sharpness:', self.sharpness)
    
    return model
  
  def create_population(self):
    for i in range(self.pop_size):
      if self.transfer:
        model = self.create_transfer_cnn()
      else:
        model = self.create_nn()
      self.models.append(model)
      self.pop.append(self.serialize(model))

      print('Model', i+1, 'created.')
  #----------------------------------------------------------------------------+

  #--- Fitness Calculation ----------------------------------------------------+

  def quality(self, model, i):
    '''
      fitness function. Returns quality of model
      Runs 1 episode of environment
    '''
    print(f'Testing model {i}...', end='')
    total_rewards = []
    for epoch in range(self.sharpness):
      self.episodes += 1
      done = False
      rewards = []
      envstate = self.env.reset()
      while not done:
        action = self.predict(model, envstate)
        envstate, reward, done, info = self.env.step(action)
        rewards.append(reward)
      
      total_rewards.append(sum(rewards))
    
    result = sum(total_rewards)/len(total_rewards)
    print(result)
    return result
  
  #----------------------------------------------------------------------------+
  
  #--- Breed Population -------------------------------------------------------+
  def selection(self):
    '''
      generate mating pool, tournament && elistist selection policy
    '''
    selection = []

    ranked = [] #ranked models, best to worst
    for i, model in enumerate(self.models):
      fitness = self.quality(model, i)
      ranked.append((i, fitness))
      if self.fitness_goal is not None and fitness >= self.fitness_goal:
        #goal met? If so, early stop
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
          for model in random.sample(ranked, len(ranked)):
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
    i = 0 #parent number, genes to get
    while len(children) < self.pop_size:
      parent1 = parents[i]
      parent2 = parents[len(parents)-i-1]

      parent1_genes = self.pop[parent1[0]]
      parent2_genes = self.pop[parent2[0]]
      if self.cxtype == 'splice':
        if self.num_layers > 1:
          genes = []
          for index, len_ in enumerate(self.weights_lengths): #splice each layer
            if index == 0:
              range_ = (0, len_)
            else:
              range_ = (self.weights_lengths[index-1], len_)

            #splice genes
            start = range_[0]
            end = range_[1]
            geneA = random.randrange(start, end)
            geneB = random.randrange(geneA, end+1)
            geneA -= start
            geneB -= start

            genes.append(splice_list(parent1_genes[start:end], parent2_genes[start:end], geneA, geneB))
          child = flatten(genes)
        else:
          geneA = random.randrange(0, len(parent1_genes))
          geneB = random.randrange(geneA, len(parent1_genes)+1)

          child = splice_list(parent1_genes, parent2_genes, geneA, geneB)
      else:
        child = ((np.array(parent1_genes) + np.array(parent2_genes)) / 2).tolist()
      
      children.append(child)
      i+=1
    
    return children
  
  def mutate(self, population):
    if self.mx_type!='default':
      '''randomize layers'''
      begin = 0
      for ind, individual in enumerate(population):
        if ind >= self.elitist:
          for i, val in enumerate(self.weights_lengths):
            if random.random() < self.mxrt:
              for gene in range(begin, val):
                individual[gene] = random.uniform(-1, 1)
            begin=val

    else:   
      for ind, individual in enumerate(population):
        if ind >= self.elitist: #ignore elites
          for i, gene in enumerate(individual):
            mxrt = self.mxrt
            if self.random_children and mxrt != 1:
              if ind == len(population) - self.random_children: #Randomly initialize last child
                mxrt = 1
            if random.random() < mxrt:
              individual[i] = random.uniform(-1, 1)
    
    return population
  #----------------------------------------------------------------------------+
  
  #--- Train/Evaluate ---------------------------------------------------------+

  def train(self, filename=None):
    self.create_population()
    print('Population created', len(self.pop))

    if filename:
      self.models[0].load_weights(filename)
      self.pop[0] = self.serialize(self.models[0])
      print('Model loaded from', filename)

    for i in range(self.generations):
      print('\nGeneration:', i+1, '/', self.generations)
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
        #create new pop
        if self.transfer:
          for i, individual in enumerate(new_pop):
            model = self.models[i]
            self.models[i] = self.create_transfer_cnn(\
              ref_model=model, fcn_weights=agents.deserialize(individual)
            )
        else: 
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


  def evaluate(self, filename=None, epochs=0):
    if self.goal_met or filename:
      #load model
      if filename:
        if self.transfer:
          model = self.create_transfer_cnn()
        else:
          model = self.create_nn()
        model.load_weights(filename)
        print(f'Weights loaded from {filename}')
      else:
        model = self.goal_met

      epoch = 0
      total_rewards = []
      #display results
      while (True, epoch<epochs)[epochs>0]:
        done = False
        rewards = []
        envstate = self.env.reset()
        while not done:
          action = self.predict(model, envstate)
          envstate, reward, done, info = self.env.step(action)
          if not epochs:
            self.env.render()
          rewards.append(reward)

        print('Reward:', sum(rewards))
        total_rewards.append(sum(rewards))
        rewards = []
        epoch+=1
      print('Epochs:', epoch, 'Average reward:', sum(total_rewards)/len(total_rewards))
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
        action = self.predict(model, envstate)
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

  def predict(self, model, envstate):
    ''' decide best action for model. '''
    qvals = model.predict(self.adj_envstate(envstate))[0] 
    if self.num_outputs == 1:
      action = qvals #continuous action space
    else:
      action = np.argmax(qvals) #discrete action space
    
    return action

  def adj_envstate(self, envstate):
    if self.transfer:
      return np.expand_dims(envstate, axis=0)
    return envstate.reshape(1, -1)

  def serialize(self, model):
    '''
      serializes model's weights into a gene string
    '''
    
    if self.transfer:
        weights = model.get_weights()[-len(self.weight_shapes):]
    else:
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
    splice = list1[index1:index2+1]
    splice += list2[index2+1:len(list1)]
  else:
    splice = list2[:index1] + list1[index1:index2+1]
    splice += list2[index2+1:len(list1)]
  
  return splice

def flatten(L):
  'flatten 2d list'
  flat = []
  for l in L:
    flat += l
  
  return flat
#------------------------------------------------------------------------------+


env = gym.make('BattleZone-v0')
print('Environment created')
# print(hasattr(env.action_space, 'n'))

config = {
  'tour': 3, 
  'cxrt': .2,
  'layers': 0, 
  'env': env, 
  'elitist': 2,
  'sharpness': 2,
  'cxtype': 'splice',
  'population': 16, 
  'mxrt': 0.00002,
  'transfer': True,
  'generations': 20, 
  'mx_type': 'default',
  'selection': 'tour',
  'fitness_goal': 6000,
  'random_children': 1,
  'validation_size': 2,
  'activation': 'linear', 
  'nodes_per_layer': [], 
}

#train model
try:
    agents = NNEvo(**config)
    agents.train('best_model.h5')
    agents.show_plot()
    agents.evaluate()
except:
    print('\nAborting...')
    agents.models[agents.best_fit[0]].save_weights('ex_model.h5')
    print('Best results saved to ex_model.h5')

#test model
# try:
#     agents = NNEvo(**config)
#     agents.evaluate('ex_model.h5')
# except:
#     env.close()