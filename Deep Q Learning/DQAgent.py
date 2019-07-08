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

        self.start_time    = datetime.datetime.now()
        self.weights_file  = kwargs.get('WEIGHTS_FILE',       "")

        # Hyperparameters
        self.batch_size    = kwargs.get('BATCH_SIZE',         8)     # How many steps (samples) to use for training
        self.max_steps     = kwargs.get('MAX_STEPS',          500)
        self.epsilon_start = kwargs.get('EPSILON_START',      0.98)
        self.discount      = kwargs.get('DISCOUNT',           0.99)
        self.replay_size   = kwargs.get('REPLAY_MEMORY_SIZE', 1000)  #steps in memory
        self.n_epoch       = kwargs.get('N_EPOCH',            15000) #number of epochs to train on
        self.min_epsilon   = kwargs.get('MIN_EPSILON',        0.001)
        self.learning_rate = kwargs.get('LEARNING_RATE',      0.001)
        
        # Data Recording Variables
        self.show_every            = kwargs.get('SHOW_EVERY',            10)
        self.aggregate_stats_every = kwargs.get('AGGREGATE_STATS_EVERY',  5)

        # Exploration settings    
        self.explore_spec = {'EPSILON_DECAY': epsilon_start,
                             'MIN_EPSILON':   min_epsilon}

        # Memory
        self.best_reward = {}
        self.memory      = list()
        self.discount    = discount
        self.max_memory  = max_memory
        self.num_actions = model.output_shape[-1]

        if model:
            self.model = model
        elif self.weights_file:
            self.model = build_model(lr = learning_rate)
            self.model = model.load_weights(weights_file)

        def build_model(self, lr = 0.001):
            pass

        def train(self):
            pass  
        
        def save_weights(self, filename):
            pass

        def predict(self): #Action decision polcicy options?
            pass

        def remember(self, episode):
            'Add to replay buffer'
            pass

        def get_batch(self):
            pass
        
        def learn(self):
            pass
        
        def evaluate(self):
            pass
        
        def load_weights(self):
            pass
