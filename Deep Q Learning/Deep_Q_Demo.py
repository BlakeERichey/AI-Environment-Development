import gym, os
from os      import path
from DQAgent import DQAgent

env = gym.make('CartPole-v0')

agent_opts = {
                'BATCH_SIZE':             16,
                'AGGREGATE_STATS_EVERY':   5, # not implemented
                'SHOW_EVERY':              2,
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .99,
                'DISCOUNT':              .90,
                'MAX_STEPS':             200,
                'MIN_EPSILON' :          .10,
                'REPLAY_MEMORY_SIZE':    500,
                'LEARNING_RATE':         0.01,
                'ACTION_POLICY':         'eg',
                'EPOCH_REWARD_GOAL':     200,
                'REWARD_GOAL':           False,
                'BEST_MODEL_FILE':       'best_model.h5',
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
#agent = DQAgent(env, **agent_opts)
#agent.build_model(**model_opts)
## agent.load_weights('cartpole.h5')
#agent.train(n_epochs=300)
#agent.save_weights('cartpole')
#agent.evaluate(n_epochs=5)
#env.close()
#print('Best Reward:', agent.best_reward)

#Evaluate models
if path.isfile('best_model.h5'):
  agent = DQAgent(env, **agent_opts)
  agent.build_model(**model_opts)
  agent.load_weights('best_model.h5')
  agent.evaluate(n_epochs=5)
