# -*- coding: utf-8 -*-
'''
Created on Monday July 8, 2019

@author: Blake Richey
'''

import gym
import os, datetime, random
import numpy          as np
import tensorflow     as tf
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.keras.optimizers import Adam
from   tensorflow.keras.callbacks  import TensorBoard
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

                
class Utilities():
    """
        Utilities for agent
        @author Siby Plathottam
    """
    import tensorflow        as tf
    import matplotlib.pyplot as plt
    """Miscelleneous utilities."""
    
    def collect_aggregate_rewards(self,episode,average_reward,min_reward,max_reward):
        """Collect rewards statistics."""
        
        print('Storing rewards @ Episode:{},Steps:{}'.format(episode,self.env.steps))       
       
        self.aggregate_episode_rewards['episode'].append(episode)
        self.aggregate_episode_rewards['average'].append(average_reward)
        self.aggregate_episode_rewards['min'].append(min_reward)
        self.aggregate_episode_rewards['max'].append(max_reward)        
        
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {self.epsilon:.3f}')        
    
    def show_plots(self):
        """Show plots."""
        
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['average'], label="average rewards")
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['max'], label="max rewards")
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['min'], label="min rewards")
        plt.legend(loc=4)
        plt.show()



class DQAgent(Utilities):

    def __init__(self, env, model=None, **kwargs):
        assert isinstance(env,gym.wrappers.time_limit.TimeLimit),\
            "Environment should be a Gym environment."

        self.env = env
        self.weights_file  = kwargs.get('WEIGHTS_FILE',       "")

        # Hyperparameters
        self.batch_size    = kwargs.get('BATCH_SIZE',         8)     # How many steps (samples) to use for training
        self.max_steps     = kwargs.get('MAX_STEPS',          500)
        self.epsilon_start = kwargs.get('EPSILON_START',      0.98)
        self.discount      = kwargs.get('DISCOUNT',           0.99)
        self.replay_size   = kwargs.get('REPLAY_MEMORY_SIZE', 1000)  #steps in memory
        self.min_epsilon   = kwargs.get('MIN_EPSILON',        0.001)
        self.learning_rate = kwargs.get('LEARNING_RATE',      0.001)
        self.action_policy = kwargs.get('ACTION_POLICY',      'softmax')
        
        # Data Recording Variables
        self.show_every            = kwargs.get('SHOW_EVERY',            10)
        self.aggregate_stats_every = kwargs.get('AGGREGATE_STATS_EVERY',  5)

        # Exploration settings    
        self.explore_spec = {'EPSILON_DECAY': self.epsilon_start,
                             'MIN_EPSILON':   self.min_epsilon}

        # Memory
        self.best_reward = {}
        self.memory      = list()

        if model:
            self.model = model
        elif self.weights_file:
            self.build_model(lr = self.learning_rate)
            self.model = model.load_weights(self.weights_file)

    def build_model(self, model_type='dense', lr = 0.001, **kwargs):
      if not hasattr(self, 'model'):
        #define NN
        self.num_outputs     = self.env.action_space.n 
        self.num_layers      = kwargs.get('num_layers',      3)
        self.default_nodes   = kwargs.get('default_nodes',   20)
        self.nodes_per_layer = kwargs.get('nodes_per_layer', [])
        self.dropout_rate    = kwargs.get('dropout_rate',    0.5)
        self.add_dropout     = kwargs.get('add_dropout',     False)
        self.activation      = kwargs.get('activation',      'softmax')
        self.num_features    = self.env.observation_space.shape[0]

        #Create NN
        if model_type == 'dense':
          assert self.num_layers >=1, 'Number of layers should be greater than or equal to one!'

          model = Sequential()
          model.add(Dense(self.num_features, input_shape = (self.num_features,)))
          
          for layer in range(self.num_layers):
  
            try:
              nodes=self.nodes_per_layer[layer]
            except IndexError:
              nodes = None

            if nodes is None:
              nodes = self.default_nodes

            print(layer)
            model.add(Dense(units = nodes, activation = 'relu'))
            print('Added Dense layer with ' + str(nodes) + ' nodes.')
            if self.add_dropout:
              model.add(Dropout(rate = self.dropout_rate, name='dropout_'+str(layer+1)))
              print('Added Dropout to layer')
          
          #output layer
          model.add(Dense(units = self.num_outputs, activation = self.activation, name='dense_output'))
          model.compile(optimizer = Adam(lr=lr), loss = 'mse', metrics=['accuracy']) #Add loss for cross entropy?
          model.summary()

      
      self.model = model
    
    def evaluate(self):
      pass

    def get_batch(self):
      pass

    def learn(self):
      pass
    
    def load_weights(self):
      pass
    
    def predict(self): #Action decision polcicy options?
      pass

    def remember(self, episode):
      'Add to replay buffer'

      envstate, action, reward, next_envstate, done = episode
      if reward > self.best_reward.get('Reward', 0):
        self.best_reward = {'Observation': next_envstate, 'Reward': reward}
      
      self.memory.append(episode)
      if len(self.memory) > self.replay_size:
        del self.memory[0]
    
    def save_weights(self, filename):
      assert self.model, 'Model must be present to save weights'
      h5file = filename + ".h5"
      self.model.save_weights(h5file, overwrite=True)
      print('Weights saved to:', h5file)
    
    def train(self, n_epochs=15000, max_steps=500):
      self.start_time    = datetime.datetime.now()
      pass