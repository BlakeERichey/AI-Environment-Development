import gym, os
from os      import path
from DQAgent import DQAgent

env = gym.make('MountainCar-v0')

agent_opts = {
                #hyperparameters
                'BATCH_SIZE':             16,
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .9992,
                'DISCOUNT':              .99,
                'MAX_STEPS':             200,
                'MIN_EPSILON' :          0.01,
                'REPLAY_MEMORY_SIZE':    800,
                'LEARNING_RATE':         0.0001,
                'ACTION_POLICY':         'eg',

                #saving and logging results
                'AGGREGATE_STATS_EVERY':   100,
                'SHOW_EVERY':              2,
                'COLLECT_RESULTS':      True,
                'COLLECT_CUMULATIVE':   True,
                'SAVE_EVERY_EPOCH':     True,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      'mountain_best.h5',
            } 

model_opts = {
                'num_layers':      2,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'add_dropout':     False,
                'add_callbacks':   False,
                'activation':      'linear',
                'nodes_per_layer': [64, 64],
            }

#Train models
agent = DQAgent(env, **agent_opts)
agent.build_model(**model_opts)
# agent.load_weights('mountain_best')
agent.train(n_epochs=3000)
agent.save_weights('mountaincar')
agent.show_plots()
env.close()
print('Best Reward:', agent.best_reward)

#Evaluate models
# if path.isfile('mountain_best.h5'):
#   agent = DQAgent(env, **agent_opts)
#   agent.build_model(**model_opts)
#   agent.load_weights('mountain_best')
#   results = agent.evaluate(n_epochs=5, render=True, verbose=True)
#   print('Average Reward over 5 epochs', sum(sum(results,[]))/len(results))
