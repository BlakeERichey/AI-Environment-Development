'''
Created on Wednesday November 11, 2019

@author: Blake Richey

Implementation of NeuroEvolution Algorithm:
  Develop a neurel network and implement genetic algorithm that finds optimum
  weights, as an alternative to backpropogation
'''

import gym, operator, time
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
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

import traceback, logging
from multiprocessing import Pool, Process, Queue, Array

#speed up forward propogation
backend.set_learning_phase(0)

#disable warnings in subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

def deserialize(genes, shapes, lengths):
  '''
    deserializes gene string into weights
  '''

  weights = []
  for i, val in enumerate(lengths):
    if i == 0:
      begin = 0
    else:
      begin = lengths[i-1]
    weights.append(np.array(genes[begin:val]).reshape(shapes[i]))

  return weights

def multi_quality(
  res=None, 
  env=None,
  layers=1,
  shapes=(1,),
  lengths=(1,), 
  inputs=None,
  outputs=1,
  genes=None,
  index=None,
  sharpness=1,
  activation='linear',
  nodes_per_layer=[128],
  transfer=False):

  '''
    implements multiprocessed nn evaluation on a gym environment
    res: results are indexed into res at `index` 
  '''
  try:
    genes = [val for val in genes]
    print(f'Testing model {index}')
    if not transfer:
      model = Sequential()
      model.add(Dense(inputs, input_shape = (inputs,)))
      
      for layer in range(layers):

          try:
              nodes=nodes_per_layer[layer]
          except IndexError:
              nodes = None

          if nodes is None:
              nodes = 128

          model.add(Dense(units = nodes, activation = 'relu'))
      
      #output layer
      model.add(Dense(units = outputs, activation = activation))
      model.compile(optimizer = Adam(lr=0.001), loss = 'mse', metrics=['accuracy'])
    elif transfer:
      model = ResNet50(weights='imagenet', include_top=False, input_shape=(env.observation_space.shape))
      for layer in model.layers:
        layer.trainable = False
      
      flattened = Flatten()(model.output)
      #Add FCN
      for layer in range(layers):

        try:
            nodes=nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = 128

        if layer == 0:
          add_layer = Dense(units = nodes, activation = 'relu')(flattened)
        else:
          add_layer = Dense(units = nodes, activation = 'relu')(add_layer)
      
      if layers:
        output = Dense(units = outputs, activation = activation)(add_layer)
      else:
        output = Dense(units = outputs, activation = activation)(flattened)

      model = Model(model.inputs, output)
      model.compile(Adam(lr=1e-3), 'mse', metrics=['acc'])

    if transfer:
      fcn_weights = deserialize(genes, shapes, lengths)
      assert len(fcn_weights) == len(shapes), \
        f'Invalid Weight Structure. Expected {len(shapes)}, got {len(fcn_weights)}.'
      all_weights = model.get_weights()
      untrainable = all_weights[:-len(shapes)]
      weights = all_weights[-len(shapes):]
      for i, matrix in enumerate(weights):
        matrix[:] = fcn_weights[i]
    
      model.set_weights(untrainable + weights)
    else:
      weights = deserialize(genes, shapes, lengths)
      model.set_weights(weights)
    
    # print('index', index)
    # print('genes', genes)
    # print('weights', weights, '\n\n\n')

    total_rewards = []
    for epoch in range(sharpness):
      done = False
      rewards = []
      envstate = env.reset()
      while not done:
        #adj envstate
        if transfer:
          envstate = np.expand_dims(envstate, axis=0)
        else:
          envstate = envstate.reshape(1, -1)
        
        qvals = model.predict(envstate)[0]
        if outputs == 1:
          action = qvals  #continuous action space
        else:
          action = np.argmax(qvals) #discrete action space

        envstate, reward, done, info = env.step(action)
        rewards.append(reward)
      
      total_rewards.append(sum(rewards))
    
    if 5 >= sharpness >= 1:
      result = max(total_rewards)
    else:
      result = sum(total_rewards)/len(total_rewards)
  except:
    print('Exception Occured in Process!')
    result = -1000000
  print(f'Model {index} Results: {result}')
  res[index] = result

  # spontaneous saving
  if index == 0:
    print(f'Saving model {index}...')
    model.save_weights('BattleZoneTemp.h5')
    print('Model saved')
  return result


class NNEvo:

  def __init__(self, 
    tour=3,
    cores=1,
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
        'cores': 1,
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

    self.default_nodes   = 128
    self.mxrt            = mxrt        #chance of a single weight being mutated
    self.cxrt            = cxrt        #chance of parent being selected (crossover rate)
    self.best_fit        = None        #(model, fitness) with best fitness
    self.tour            = tour        #tournament sample size when using tour selection policy
    self.cores           = cores       #how many cores to run forward propogation on
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
    
    #create environments
    self.envs = [gym.make(env) for _ in range(self.cores)]
    print('Environments Created:', self.cores)
    self.num_features = self.envs[0].observation_space.shape[0]

    
    outputs = 1
    if hasattr(self.envs[0].action_space, 'n'):
      outputs = self.envs[0].action_space.n
    self.num_outputs     = outputs

    self.models = [] #list of individuals 
    self.pop    = [] #population (2d-list of weights)
    self.weight_shapes   = None
    self.weights_lengths = None
    self.plots = [] #points for matplotlib
    self.episodes = 0

    self.best_results = {}

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
      model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.envs[0].observation_space.shape))
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

      arr: subprocess accessible memory for storing results. indexed by model number `i`
    '''
    print(f'Testing model {i}...', end='')
    total_rewards = []
    for epoch in range(self.sharpness):
      self.episodes += 1
      done = False
      rewards = []
      envstate = self.envs[0].reset()
      while not done:
        action = self.predict(model, envstate)
        envstate, reward, done, info = self.envs[0].step(action)
        rewards.append(reward)
      
      total_rewards.append(sum(rewards))
    
    if 5 >= self.sharpness >= 1:
      result = max(total_rewards)
    else:
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
    if self.best_fit[1] >= self.best_results.get('fitness', -1000000):
      self.best_results['fitness'] = self.best_fit[1]
      self.best_results['genes'] = [gene for gene in self.pop[self.best_fit[0]]]
    return selection

  def selection_mp(self):
    '''
      runs processes in parallel
      generate mating pool, tournament && elistist selection policy
    '''
    selection = []

    #create gene dependencies
    all_genes = []
    for i in range(self.pop_size):
      genes = Array('f', range(len(self.pop[i])))
      for j in range(len(self.pop[i])):
        genes[j] = self.pop[i][j]
      all_genes.append(genes)

    nodes_per_layer = Array('f', range(len(self.nodes_per_layer)))
    for j in range(len(self.nodes_per_layer)):
      nodes_per_layer[j] = self.nodes_per_layer[j]

    fitnesses = Array('f', range(self.pop_size))
    processed = 0
    processes = []
    while processed < self.pop_size:
      if processed < self.pop_size - len(processes) and  len(processes) < self.cores:
        i = len(processes) + processed

        genes = all_genes[i]
        
        obj = {
          'index':      i,
          'genes':      genes,
          'res':        fitnesses, 
          'env':        self.envs[len(processes)], 
          'layers':     self.num_layers,
          'transfer':   self.transfer,
          'outputs':    self.num_outputs,
          'inputs':     self.num_features,
          'sharpness':  self.sharpness,
          'shapes':     self.weight_shapes,
          'activation': self.activation,
          'lengths':    self.weights_lengths,
          'nodes_per_layer': nodes_per_layer,
        }

        # print('Restructred Genes', obj['index'], [val for val in genes], '\n\n\n')
        p = Process(target=multi_quality, kwargs=obj)
        processes.append(p)
        p.start()
      
      #remove completed processes
      ind = 0
      while ind < len(processes):
        p = processes[ind]
        if not p.is_alive():
          #terminate process
          p.join()
          processes.pop(ind)
          ind -= 1

          processed += 1
          self.episodes += self.sharpness
        ind += 1

    ranked = [] #ranked models, best to worst
    results = [val for val in fitnesses]
    for i, fitness in enumerate(results):
      ranked.append((i, fitness))

    ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)
    print('Ranked:', ranked)
    self.best_fit = ranked[0]
    
    for model in ranked: #model = (i, fitness)
      if self.fitness_goal is not None and model[1] >= self.fitness_goal:
        #goal met? If so, early stop
        i = model[0] #model number
        if self.validation_size:
          valid = self.validate(self.models[i])
        else:
          valid = True
        
        if valid:
          self.goal_met = self.models[i] #save model that met goal
          self.best_fit = model
          break

    if not self.goal_met:  #if goal met prepare to terminate
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
    if self.best_fit[1] >= self.best_results.get('fitness', -1000000) or self.goal_met:
      self.best_results['fitness'] = self.best_fit[1]
      self.best_results['genes'] = [gene for gene in self.pop[self.best_fit[0]]]
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

  def train(self, filename=None, target='best_model.h5'):
    self.create_population()
    print('Population created', len(self.pop))

    if filename:
      self.models[0].load_weights(filename)
      self.pop[0] = self.serialize(self.models[0])
      print('Model loaded from', filename)

    for i in range(self.generations):
      print('\nGeneration:', i+1, '/', self.generations)
      if self.cores > 1:
        parents = self.selection_mp()
      else:
        parents = self.selection()

      if i == self.generations - 1:
          break
      if not self.goal_met:
        print('Goal not met. Parents selected.')
        print('Best fit:', self.best_fit)
        print('Best Results', self.best_results.get('fitness'))
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
        self.goal_met.save_weights(target)
        print(f'Best results saved to {target}')
        break
    
    self.save_best(target=target)


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
        envstate = self.envs[0].reset()
        while not done:
          action = self.predict(model, envstate)
          envstate, reward, done, info = self.envs[0].step(action)
          if not epochs:
            self.envs[0].render()
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
      envstate = self.envs[0].reset()
      while not done:
        action = self.predict(model, envstate)
        envstate, reward, done, info = self.envs[0].step(action)
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

  def save_best(self, target='best_model.h5'):
    if not self.goal_met:
      if self.best_results['fitness']:
        genes = self.best_results['genes']
        model = self.models[0]
        if self.transfer:
          self.models[0] = self.create_transfer_cnn(\
            ref_model=model, fcn_weights=agents.deserialize(genes)
          )
        else:
          model.set_weights(self.deserialize(genes))
        model.save_weights(target)
      elif self.best_fit:
        self.models[self.best_fit[0]].save_weights(target)
      print(f'Best results saved to {target}')

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

if __name__ == '__main__':
  #train model
  try:
    agents = NNEvo(**config)
    agents.train(filename='BattleZoneTemp.h5', target='BattleZone.h5')
    agents.show_plot()
    agents.evaluate()
  except:
    traceback.print_exc()
    print('\nAborting...')
    agents.save_best(target='ex_model_battlezone.h5')
    print('Best results saved to ex_model_battlezone.h5')

#  test model
  # try:
  #     agents = NNEvo(**config)
  #     agents.evaluate('MountainCar.h5')
  # except:
  #     traceback.print_exc()
  #     agents.envs[0].close()
