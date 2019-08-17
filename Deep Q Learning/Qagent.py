import numpy as np
import os, datetime, random, gym, json, pickle
class Qagent():

  def __init__(self, env):
    self.discount = 0.9
    self.epsilon = 1
    self.learning_rate = 0.001
    self.print_every   = 100
    self.min_epsilon = 0.00
    if env:
        self.build_model(env)

  def build_model(self, env):
    self.q_table = {}
    self.env = env
  
  def predict(self, envstate):
    envstate = tuple(envstate.reshape(1, -1).tolist()[0])
    envstate = self.adj_state(envstate)
    
    if np.random.rand() < self.epsilon:
      action = random.choice(range(self.env.action_space.n))
    else:
      qvals = self.q_table.get(envstate)
      if not hasattr(qvals, 'shape'):
        qvals = self.populate(envstate)
      action = np.argmax(qvals)
    
    return action
  
  def populate(self, envstate):
    qvals = np.array([random.uniform(-1,1) for _ in range(self.env.action_space.n)])
    self.q_table[envstate] = qvals
    return qvals

  def learn(self, episode):
    prev_envstate, action, reward, envstate, done = episode
    envstate = tuple(envstate.reshape(1, -1).tolist()[0])
    prev_envstate = tuple(prev_envstate.reshape(1, -1).tolist()[0])

    envstate = self.adj_state(envstate)
    prev_envstate = self.adj_state(prev_envstate)

    qvals = self.q_table.get(prev_envstate)
    if not hasattr(qvals, 'shape'):
        qvals = self.populate(prev_envstate)

    qvals_future = self.q_table.get(envstate)
    if not hasattr(qvals_future, 'shape'):
        qvals_future = self.populate(envstate)
    qval = qvals[action]
    
    if not done:
      new_q = (1-self.learning_rate)*qval + self.learning_rate * (reward + self.discount * max(qvals_future))
    else:
      new_q = reward
    
    # print('qvals', qvals)
    qvals[action] = new_q
    # print('new qvals', qvals)
    self.q_table[prev_envstate] = qvals


  def train(self, n_epochs=15000, max_steps=0):
    self.epsilon_decay = (self.epsilon - self.min_epsilon)/n_epochs

    for epoch in range(n_epochs):
      envstate = self.env.reset()
      done = False
      rewards = []
      while not done:
        prev_envstate = envstate
        action = self.predict(envstate)
        envstate, reward, done, info = self.env.step(action)
        
        episode = [prev_envstate, action, reward, envstate, done] #reward player 1
        self.learn(episode)
        # self.env.render()
        
        rewards.append(reward)
      
      #end of epoch
      if epoch % self.print_every == 0:
        print(f'{epoch}/{n_epochs} Epochs complete | Reward: {sum(rewards)} | Epsilon {self.epsilon}')
      rewards = []
      self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

  def save(self, filename):
    with open (filename, 'wb') as f:
      pickle.dump(self.q_table, f)
  
  def load(self, filename):
    with open(filename, 'rb') as f:
      self.q_table = pickle.load(f)

  def evaluate(self, n_epochs = 1, render=True):
    total_rewards = []
    for epoch in range(n_epochs):
      envstate = self.env.reset()
      done = False
      rewards = []
      while not done:
        envstate = tuple(envstate.reshape(1, -1).tolist()[0])
        envstate = self.adj_state(envstate)
        qvals = self.q_table.get(envstate)
        if not hasattr(qvals, 'shape'):
          qvals = self.populate(envstate)
#        print(envstate, qvals)
        action = np.argmax(qvals)
        envstate, reward, done, info = self.env.step(action)

        rewards.append(reward)
        
        if render:
          self.env.render()

      if epoch % 1 == 0:
        print(f'{epoch}/{n_epochs} Epochs complete')
      total_rewards.append(rewards)
    
    return total_rewards

  def adj_state(self, envstate):
    adj = [truncate_number(i, n_decimals=2) for i in envstate]
    return tuple(adj)

#removes imprecision issues with floats: 1-.1 = .900000000001 => 1-.1 = .9
def truncate_number(f_number, n_decimals=9):
  strFormNum = "{0:." + str(n_decimals+5) + "f}"
  trunc_num  = float(strFormNum.format(f_number)[:-5])
  return(trunc_num)

def save_json(obj, filename):
  '''save obj to filename. obj is expected to be a json format'''
  with open(filename, 'w+') as f:
    json.dump(obj, f)

def load_json(filename):
  '''returns dictionary with json data'''
  with open(filename, 'r') as f:
    obj = json.loads(f.read())
  
    return obj

# Example use
# env = gym.make('MountainCar-v0')
# agent = Qagent(env)
# agent.train(5000)
# agent.save('mountaincar.pickle')
# agent.evaluate()
# env.close()