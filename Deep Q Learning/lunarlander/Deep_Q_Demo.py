import gym, os
from os      import path
from DQAgent import DQAgent

env = gym.make('LunarLander-v2')

agent_opts = {
                #hyperparameters
                'BATCH_SIZE':             16,
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .99,
                'DISCOUNT':              .90,
                'MAX_STEPS':             500,
                'MIN_EPSILON' :          0.100,
                'REPLAY_MEMORY_SIZE':    1000,
                'LEARNING_RATE':         0.01,
                'ACTION_POLICY':         'eg',

                #saving and logging results
                'AGGREGATE_STATS_EVERY':   25,
                'SHOW_EVERY':              2,
                'COLLECT_RESULTS':      True,
                'COLLECT_CUMULATIVE':   True,
                'SAVE_EVERY_EPOCH':     True,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      'LunarLander_best.h5',
            } 

model_opts = {
                'num_layers':      1,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'add_dropout':     False,
                'add_callbacks':   False,
                'activation':      'linear',
                'nodes_per_layer': [10],
            }

#Train models
agent = DQAgent(env, **agent_opts)
agent.build_model(**model_opts)
# agent.load_weights('mountain_best')
agent.train(n_epochs=500)
agent.save_weights('lunarlander')
agent.show_plots()
env.close()
print('Best Reward:', agent.best_reward)

#Evaluate models
if path.isfile('LunarLander_best.h5.h5'):
  agent = DQAgent(env, **agent_opts)
  agent.build_model(**model_opts)
  agent.load_weights('LunarLander_best.h5')
  results = agent.evaluate(n_epochs=5, render=True, verbose=True)
  print('Average Reward over 5 epochs', sum(sum(results,[]))/len(results))
