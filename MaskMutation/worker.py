import math, random
import numpy as np

class Worker:

  def __init__(self, network, mutations=1, patience=25, alpha=.05):
    '''
      mutations: number of layers mutated on average
    '''

    self.net = network
    self.alpha = alpha
    self.patience = patience
    self.mutations = mutations #layers to mutate on average
    
    self.history = [] #history of priorities
    self.gen_mask()

  def fitness(self, env, episodes, validate=False, render=False):
    continuous = True
    if hasattr(env, "action_space.n"):
      print("Discrete Environment")
      continuous = False

    total_rewards = []
    for epoch in range(episodes):
      done = False
      rewards = []
      envstate = env.reset()
      while not done:
        action = self.predict(nn, envstate, continuous)
        envstate, reward, done, info = env.step(action)
        rewards.append(reward)
        if render:
          env.render()
      
      total_rewards.append(sum(rewards))
    
    result = round(sum(total_rewards)/len(total_rewards), 5)

    total_rewards = []
    if validate:
      #validate model
      for epoch in range(episodes):
        done = False
        rewards = []
        envstate = env.reset()
        while not done:
          action = self.predict(nn, envstate, continuous)
          envstate, reward, done, info = env.step(action)
          rewards.append(reward)
          if render:
            env.render()
        
        total_rewards.append(sum(rewards))

    validation_results = round(sum(total_rewards)/len(total_rewards), 5)

    return result, validation_results

  def breed(self,worker):
    num_layers = len(worker.net.layers)
    splice = np.random.randint(2, size=num_layers)
    new_weights = []
    for i, layer in enumerate(worker.net.layers):
      if splice[i] == 0:
        new_weights.append(np.copy(self.net.layers[i].weights))
      else:
        new_weights.append(np.copy(layer.weights))
    
    return new_weights
  
  def mutate(self,):
    for i, layer in enumerate(self.net.layers):
      if np.random.uniform() < self.mutations / len(self.net.layers):
        print("Mutating", i)
        rows, cols = layer.rows, layer.cols
        layer.weights = np.add(layer.weights, self.mask[:rows, :cols])

  def add_rank(self, rank, pop_size):
    '''takes rank in a generation and logs it to history '''

    self.history.append(rank)
    if len(self.history)>self.patience:
      del self.history[0]
    
    low_performing = 0
    thresh = int(.75*pop_size)
    for i, val in range(self.history):
      if val >= thresh:
        low_performing+=1
    
    if low_performing >= self.patience:
      self.gen_mask()

  def gen_mask(self,):
    #GENERATE MASK
    largest_layer = 0 #index of largest layer
    params = 0 #number of params in largest layer
    for i, layer in enumerate(self.net.layers):
      if layer.weights.size > params:
        params = layer.weights.size
        largest_layer = i

    rows, cols = self.net.layers[largest_layer].rows, self.net.layers[largest_layer].cols
    limit = math.sqrt(6/(rows+cols)) #glorot uniform
    self.mask = self.alpha*np.random.uniform(low=-limit, high=limit, size=(rows*cols,)).reshape((rows, cols))

  def predict(self, model, envstate, continuous=False):
    ''' decide best action for model. utility function. '''
    qvals = model.predict(envstate)[0] 
    if continuous == True:
      action = qvals #continuous action space
    else:
      action = np.argmax(qvals) #discrete action space
    
    return action