# -*- coding: utf-8 -*-
'''
Created on Monday July 8, 2019

@author: Blake Richey
'''

import cProfile, pstats, io

import gym
import os, datetime, random
import numpy          as np
import tensorflow     as tf
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.keras.optimizers import Adam
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.compat.v1.set_random_seed(1)

                
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
    
    # This is a small utility for printing readable time strings:
    def format_time(self, seconds):
        if seconds < 400:
            s = float(seconds)
            return "%.1f seconds" % (s,)
        elif seconds < 4000:
            m = seconds / 60.0
            return "%.2f minutes" % (m,)
        else:
            h = seconds / 3600.0
            return "%.2f hours" % (h,)



class DQAgent(Utilities):

    def __init__(self, env, model=None, **kwargs):
        '''
            Initialize agent hyperparameters

            agent_opts = {
                'BATCH_SIZE':              8,
                'AGGREGATE_STATS_EVERY':   5,
                'SHOW_EVERY':             10,
                'EPSILON_START':         .98,
                'EPSILON_DECAY':         .98,
                'DISCOUNT':              .99,
                'MAX_STEPS':             500,
                'MIN_EPSILON' :          0.01,
                'REPLAY_MEMORY_SIZE':    1000,
                'LEARNING_RATE':         0.001,
                'ACTION_POLICY':         'eg',
                'EPOCH_REWARD_GOAL':     False,
                'REWARD_GOAL':           False,
                'BEST_MODEL_FILE':       'best_model.h5',
            } 
        '''
        assert isinstance(env,gym.wrappers.time_limit.TimeLimit),\
            "Environment should be a Gym environment."

        self.env = env
        self.weights_file  = kwargs.get('WEIGHTS_FILE',       "")

        # Hyperparameters
        self.batch_size        = kwargs.get('BATCH_SIZE',         8)     # How many steps (samples) to use for training
        self.max_steps         = kwargs.get('MAX_STEPS',          500)
        self.action_policy     = kwargs.get('ACTION_POLICY',      'eg')  #epsilon greedy
        self.epsilon           = kwargs.get('EPSILON_START',      0.98)
        self.epsilon_decay     = kwargs.get('EPSILON_DECAY',      0.98)
        self.discount          = kwargs.get('DISCOUNT',           0.99)  #HIGH VALUE = SHORT TERM MEMORY
        self.replay_size       = kwargs.get('REPLAY_MEMORY_SIZE', 1000)  #steps in memory
        self.min_epsilon       = kwargs.get('MIN_EPSILON',        0.01)
        self.learning_rate     = kwargs.get('LEARNING_RATE',      0.001)
        self.epoch_reward_goal = kwargs.get('EPOCH_REWARD_GOAL',  False) # Goal for entire epoch 
        self.reward_goal       = kwargs.get('REWARD_GOAL',        False) # single reward goal 
        self.best_model_file   = kwargs.get('BEST_MODEL_FILE',    'best_model.h5') #file to save best model to
        
        # Data Recording Variables
        self.show_every            = kwargs.get('SHOW_EVERY',            10)
        self.aggregate_stats_every = kwargs.get('AGGREGATE_STATS_EVERY',  5)

        # Exploration settings    

        self.explore_spec = {'EPSILON_DECAY': self.epsilon_decay,
                             'MIN_EPSILON':   self.min_epsilon}

        # Memory
        self.best_reward = {}
        self.memory      = list()

        if model:
            self.model = model
        elif self.weights_file:
            self.build_model()
            self.model = model.load_weights(self.weights_file)

    def build_model(self, model_type='dense', **kwargs):
        '''
            Builds model to be trained

            model_opts = {
                'num_layers':      3,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'add_dropout':     False,
                'add_callbacks':   False,
                'activation':      'linear',
                'nodes_per_layer': [20,20,20],
            }
        '''
        if not hasattr(self, 'model'):
            #define NN
            self.num_outputs     = self.env.action_space.n 
            self.num_layers      = kwargs.get('num_layers',      3)
            self.default_nodes   = kwargs.get('default_nodes',   20)
            self.nodes_per_layer = kwargs.get('nodes_per_layer', [])
            self.dropout_rate    = kwargs.get('dropout_rate',    0.5)
            self.add_dropout     = kwargs.get('add_dropout',     False)
            self.add_callbacks   = kwargs.get('add_callbacks',   False)
            self.activation      = kwargs.get('activation',      'linear')
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
                    print(f'Added Dense layer with {nodes} nodes.')
                    if self.add_dropout:
                        model.add(Dropout(rate = self.dropout_rate, name='dropout_'+str(layer+1)))
                        print('Added Dropout to layer')
                
                #output layer
                model.add(Dense(units = self.num_outputs, activation = self.activation, name='dense_output'))
                model.compile(optimizer = Adam(lr=self.learning_rate), loss = 'mse', metrics=['accuracy']) #Add loss for cross entropy?
                model.summary() 

        
        self.model = model
    
    def evaluate(self, n_epochs=1):
        start_time = datetime.datetime.now()
        print(f'Evaluating... Starting at: {start_time}')
        
        for epoch in range(n_epochs):
            n_steps = 0
            done = False
            envstate = self.env.reset()
            rewards = []
            while (not done and n_steps < self.max_steps):
                prev_envstate = envstate
                q             = self.model.predict(prev_envstate.reshape(1, -1))
                action        = np.argmax(q[0])
                envstate, reward, done, info = self.env.step(action)
                
                n_steps += 1
                rewards.append(reward)
                self.env.render()
            
            dt = datetime.datetime.now() - start_time
            t = self.format_time(dt.total_seconds())
            results = f'Epoch: {epoch}/{n_epochs-1} | Steps {n_steps} | Cumulative Reward: {sum(rewards)} | Time: {t}'
            print(results)
      
        self.env.close()

    def get_batch(self):
        '''
            Gets previous states to perform a batch fitting
        '''
        mem_size   = len(self.memory)
        batch_size = min(mem_size, self.batch_size)
        env_size   = self.memory[0][0].reshape(1, -1).shape[1]

        inputs = np.zeros((batch_size, env_size))
        targets = np.zeros((batch_size, self.num_outputs))
        for i, j in enumerate(np.random.choice(range(mem_size), batch_size, replace=False)):
            envstate, action, reward, next_envstate, done, target, Q_sa = self.memory[j]
            inputs[i] = envstate
            # targets[i] = self.model.predict(envstate.reshape(1, -1))
            targets[i] = target
            # Q_sa = np.max(self.model.predict(next_envstate.reshape(1, -1)))
            if done:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

    def learn(self): #add callback options?
      inputs, targets = self.get_batch()
      test_input, test_target = self.get_batch()
      
      callbacks = []
      if self.add_callbacks:
        callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]

      history = self.model.fit(
          inputs,
          targets,
          callbacks = callbacks,
          batch_size = self.batch_size//4,
          verbose=0,
      )
      loss = self.model.evaluate(inputs, targets, verbose=0)[0]

      return loss
    
    def load_weights(self, filename):
      '''loads weights from a file'''
      self.model.load_weights(filename)
      print(f'Successfully loaded weights from: {filename}')
    
    def predict(self, envstate): 
        '''
            envstate: envstate to be evaluated
            returns:  given envstate, returns best action model believes to take
        '''
        assert self.model, 'Model must be present to make prediction'

        if self.action_policy == 'softmax':
            qvals = self.model.predict(envstate.reshape(1, -1))[0]
            action = np.argmax(np.random.multinomial(1, qvals))
        elif self.action_policy == 'eg': #epsilon greedy
            if np.random.rand() < self.epsilon:
                action = random.choice(range(self.env.action_space.n))
            else:
                qvals = self.model.predict(envstate.reshape(1, -1))[0]
                action = np.argmax(qvals)

        return action


    def remember(self, episode):
      'Add to replay buffer'

      envstate, action, reward, next_envstate, done = episode
      target = self.model.predict(envstate.reshape(1, -1))
      Q_sa   = np.max(self.model.predict(next_envstate.reshape(1, -1)))
      if reward > self.best_reward.get('Reward', 0):
        self.best_reward = {'Observation': next_envstate, 'Reward': reward}
      
      self.memory.append(episode + [target, Q_sa])
      if len(self.memory) > self.replay_size:
        del self.memory[0]
    
    def save_weights(self, filename):
      assert self.model, 'Model must be present to save weights'
      h5file = filename + ".h5"
      self.model.save_weights(h5file, overwrite=True)
      print('Weights saved to:', h5file)
    
    def train(self, n_epochs=15000, max_steps=0):
        self.start_time    = datetime.datetime.now()
        print(f'Starting training at {self.start_time}')
        print(f'Action Decision Policy: {self.action_policy}')

        max_steps = max_steps or self.max_steps
        for epoch in range(n_epochs):
            n_steps  = 0
            done     = False
            envstate = self.env.reset()
            rewards = []
            while (not done and n_steps<max_steps):
                prev_envstate = envstate
                action = self.predict(prev_envstate)

                envstate, reward, done, info = self.env.step(action)

                episode = [prev_envstate, action, reward, envstate, done]
                self.remember(episode)

                loss = self.learn() #fit model
                rewards.append(reward)
                n_steps += 1

                #save model if desired goal is met
                if self.reward_goal and reward >= self.reward_goal:
                    if hasattr(self, 'best_model') and loss < self.best_model['loss']:
                        self.model.save_weights(self.best_model_file, overwrite=True)
                    else:
                        self.model.save_weights(self.best_model_file, overwrite=True)
                        self.best_model = {
                            'weights': self.model.get_weights(),
                            'loss':    loss,
                            }

            dt = datetime.datetime.now() - self.start_time
            t  = self.format_time(dt.total_seconds())
            if epoch % self.show_every == 0:
                results = f'Epoch: {epoch}/{n_epochs-1} | ' +\
                    f'Loss: %.4f | ' % loss +\
                    f'Steps {n_steps} | ' +\
                    f'Epsilon: %.3f | ' % self.epsilon +\
                    f'Time: {t}'
                print(results)
            
            #save model if desired goal is met
            if self.epoch_reward_goal and sum(rewards) >= self.epoch_reward_goal:
                if hasattr(self, 'best_model') and loss < self.best_model['loss']:
                        self.model.save_weights(self.best_model_file, overwrite=True)
                else:
                    self.model.save_weights(self.best_model_file, overwrite=True)
                    self.best_model = {
                        'weights': self.model.get_weights(),
                        'loss':    loss,
                        }

            #decay epsilon after each epoch
            if self.action_policy == 'eg':
                decay = self.explore_spec['EPSILON_DECAY']
                self.epsilon = max(self.min_epsilon, decay*self.epsilon)
    
    # def temp_get_batch(self):
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     retval = self.get_batch()
    #     pr.disable()
    #     s = io.StringIO()
    #     sortby = 'cumulative'
    #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #     ps.print_stats(.05)
    #     print(s.getvalue())
    #     return retval