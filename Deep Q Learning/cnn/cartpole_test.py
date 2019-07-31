    
import gym, os
import image_env
from os      import path
from agent.DQAgent import DQAgent
from BRprofiler import profile

env = gym.make('CartPole-v0')

agent_opts = {
                #hyperparameters
                'REPLAY_BATCH_SIZE':      32,
                'LEARNING_BATCH_SIZE':     4,
                'DISCOUNT':              .90,
                'MAX_STEPS':             200,
                'REPLAY_MEMORY_SIZE':    500,
                'LEARNING_RATE':         0.01,
                
                #ann specific
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .98,
                'MIN_EPSILON' :          0.1,

                #saving and logging results
                'AGGREGATE_STATS_EVERY':  5,
                'SHOW_EVERY':             2,
                'COLLECT_RESULTS':      True,
                'COLLECT_CUMULATIVE':   True,
                'SAVE_EVERY_EPOCH':     True,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      './agent/results/cartpole_best.h5',
            } 

model_opts = {
                'num_layers':      2,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'model_type':      'ann',
                'add_dropout':     False,
                'add_callbacks':   False,
                'nodes_per_layer': [10,10],

                #cnn options
                'filter_size':     3,
                'pool_size':       2,
                'stride_size':     None,
            }

# Train models
def train_model(agent_opts, model_opts):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    # agent.load_weights('./agent/results/best_model')
    agent.train(n_epochs=200, render=False)
    agent.save_weights('./agent/results/cartpole')
    agent.show_plots()
    env.close()

#Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    if best_model:
      filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
      agent.load_weights(filename)
    else:
      agent.load_weights('./agent/results/cartpole')
    results = agent.evaluate(50, render=False)
    print(sum(sum(results,[]))/len(results))

#train_model(agent_opts, model_opts)
evaluate_model(agent_opts, model_opts, best_model=False)