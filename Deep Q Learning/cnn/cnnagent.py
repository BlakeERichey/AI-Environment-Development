    
import gym, os
import image_env
from os      import path
from agent.DQAgent import DQAgent
from BRprofiler import profile

env = gym.make('image_env-v0')

root_path = './agent/results/'

agent_opts = {
                #hyperparameters
                'REPLAY_BATCH_SIZE':      32,
                'LEARNING_BATCH_SIZE':     4,
                'DISCOUNT':              .99,
                'MAX_STEPS':             500,
                'REPLAY_MEMORY_SIZE':    128,
                'LEARNING_RATE':         0.005,
                
                #ann specific
                'EPSILON_START':         .98,
                'EPSILON_DECAY':         .98,
                'MIN_EPSILON' :          0.01,

                #saving and logging results
                'AGGREGATE_STATS_EVERY':  5,
                'SHOW_EVERY':             25,
                'COLLECT_RESULTS':      True,
                'SAVE_EVERY_EPOCH':     False,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      f'{root_path}best_model_rnn.h5',
            } 

model_opts = {
                'num_layers':      3,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'model_type':      'cnn',
                'add_dropout':     False,
                'add_callbacks':   False,
                'nodes_per_layer': [64,32,32],

                #cnn options
                'filter_size':     3,
                'pool_size':       2,
                'stride_size':     None,
    
                #rnn options, only available for cnns
                'rnn_hidden_layers':     3,
                'node_per_hidden_layer': [20, 20, 20]
            }

# Train models
def train_model(agent_opts, model_opts):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    # agent.load_weights('./agent/results/best_model')
    agent.train(n_epochs=300, render=False)
    agent.save_weights('./agent/results/cnnagent')
    agent.show_plots()
    agent.show_plots('cumulative')
    agent.show_plots('accuracy')
    env.close()

#Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    if best_model:
      filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
      agent.load_weights(filename)
    else:
      agent.load_weights('./agent/results/cnnagent')
    results = agent.evaluate(50, render=False)
#    print(sum(sum(results,[]))/len(results))

train_model(agent_opts, model_opts)
# evaluate_model(agent_opts, model_opts, best_model=True)