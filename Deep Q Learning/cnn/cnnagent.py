    
import gym, os
import image_env
from os      import path
from DQAgent import DQAgent

env = gym.make('image_env-v0')

agent_opts = {
                #hyperparameters
                'BATCH_SIZE':             16,
                'DISCOUNT':              .90,
                'MAX_STEPS':             60000,
                'REPLAY_MEMORY_SIZE':    1000,
                'LEARNING_RATE':         0.001,
                
                #ann specific
                'EPSILON_START':         .98,
                'EPSILON_DECAY':         .98,
                'MIN_EPSILON' :          0.01,

                #saving and logging results
                'AGGREGATE_STATS_EVERY':  5,
                'SHOW_EVERY':             1,
                'COLLECT_RESULTS':      False,
                'COLLECT_CUMULATIVE':   False,
                'SAVE_EVERY_EPOCH':     False,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      'best_model.h5',
            }

model_opts = {
                'num_layers':      2,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'model_type':      'cnn',
                'add_dropout':     False,
                'add_callbacks':   False,
                'nodes_per_layer': [64, 32],

                #cnn options
                'filter_size':     3,
                'pool_size':       2,
                'stride_size':     None,
            }

agent = DQAgent(env, **agent_opts)
agent.build_model(**model_opts)
agent.load_weights('numbers2')
agent.train(n_epochs=1)
# agent.evaluate(n_epochs=1)
agent.save_weights('numbers2')
# agent.show_plots()
env.close()