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

  def fitness(self, env, episodes=1, validate=False, render=False):
    continuous = True
    if hasattr(env.action_space, "n"):
      # print("Discrete Environment")
      continuous = False

    total_rewards = []
    for epoch in range(episodes):
      done = False
      rewards = []
      envstate = env.reset()
      while not done:
        action = self.predict(self.net, envstate, continuous)
        envstate, reward, done, info = env.step(action)
        rewards.append(reward)
        if render:
          env.render()
      if render:
        print("Reward:", sum(rewards))
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
          action = self.predict(self.net, envstate, continuous)
          envstate, reward, done, info = env.step(action)
          rewards.append(reward)
          if render:
            env.render()
        
        total_rewards.append(sum(rewards))

    if not len(total_rewards):
      total_rewards = [0]
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
        rows, cols = layer.rows, layer.cols
        mask = self.mask[i]
        layer.weights = np.add(layer.weights, mask[:rows, :cols])

  def add_rank(self, rank, pop_size):
    '''takes rank in a generation and logs it to history '''

    self.history.append(rank)
    if len(self.history)>self.patience:
      del self.history[0]
    
    low_performing = 0
    thresh = int(.75*pop_size)
    for _, val in enumerate(self.history):
      if val >= thresh:
        low_performing+=1
    
    if low_performing >= self.patience:
      self.gen_mask()
      self.net = self.net.clone()

  def gen_mask(self,):
    #GENERATE MASKS
    self.mask = []
    for i, layer in enumerate(self.net.layers):
      rows, cols = layer.rows, layer.cols
      limit = math.sqrt(6/(rows+cols)) #glorot uniform
      mask = self.alpha*np.random.uniform(low=-limit, high=limit, size=(rows*cols,)).reshape((rows, cols))
      self.mask.append(mask)

  def predict(self, model, envstate, continuous=False):
    ''' decide best action for model. utility function. '''
    qvals = model.feed_forward(envstate)
    if continuous == True:
      action = qvals #continuous action space
    else:
      action = np.argmax(qvals) #discrete action space
    
    return action