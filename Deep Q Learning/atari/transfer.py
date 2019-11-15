#Implement transfer learning to identify disease presence in corn
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import axes
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.datasets import cifar10
from keras.models import Model
from keras import layers
from tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

import gym, os
from agent import DQAgent

dim = 192 #dim x dim images

#--------------- Build NN ---------------
pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(210, 160, 3))
#Set Resnet to non trainable
for layer in pretrained.layers:
  layer.trainable = False

#add FCN  
flattened = layers.Flatten()(pretrained.output)
add_layer = layers.Dense(2, activation='relu')(flattened)
add_layer = layers.Dense(64, activation='relu')(add_layer)
add_layer = layers.Dropout(rate=0.2)(add_layer)
add_layer = layers.Dense(32, activation='relu')(add_layer)
output = layers.Dense(18, activation='softmax', name='output')(add_layer)
pretrained = Model(pretrained.inputs, output)
pretrained.summary()



pretrained.compile(RMSprop(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])\

print('Input shape:', pretrained.input_shape)
print('Output shape:', pretrained.output_shape)

#--------------- Train Model ---------------

env = gym.make('BattleZone-v0')
root_path = './'
agent_opts = {
                #hyperparameters
                'REPLAY_BATCH_SIZE':      128,
                'LEARNING_BATCH_SIZE':     64,
                'DISCOUNT':               .90,
                'MAX_STEPS':             10000,
                'REPLAY_MEMORY_SIZE':    3000,
                'LEARNING_RATE':         0.001,
                
                #ann specific
                'EPSILON_START':         .98,
                'EPSILON_DECAY':         .98,
                'MIN_EPSILON' :          0.001,

                #saving and logging results
                'AGGREGATE_STATS_EVERY':  1,
                'SHOW_EVERY':             1,
                'COLLECT_RESULTS':      True,
                'SAVE_EVERY_EPOCH':     True,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      f'{root_path}best_model.h5',
            }           

model_opts = {
                'num_layers':      3,
                'default_nodes':   20,
                'dropout_rate':    0.1,
                'model_type':      'cnn',
                'add_dropout':     True,
                'add_callbacks':   False,
                'nodes_per_layer': [128,64,32],

                #cnn options
                'filter_size':     3,
                'pool_size':       2,
                'stride_size':     2,
            }

# Train models
def train_model(agent_opts, model_opts):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    agent.model = pretrained
    agent.action_policy='eg'
    agent.model_type='cnn'
    agent.load_weights(f'{root_path}best_model')
    agent.model.summary()
    agent.train(n_epochs=50, render=False)
    agent.save_weights(f'{root_path}battle')
    agent.show_plots('cumulative')
    agent.show_plots('loss')
    env.close()

#Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    agent.model=pretrained
    agent.action_policy='eg'
    agent.model_type='cnn'
    if best_model:
      filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
      agent.load_weights(filename)
    else:
      agent.load_weights(f'{root_path}racing2')
    results = agent.evaluate(1, render=True, verbose=True)
    print(f'Average Results: {sum(sum(results,[]))/len(results)}')

train_model(agent_opts, model_opts)
#evaluate_model(agent_opts, model_opts, best_model=True)