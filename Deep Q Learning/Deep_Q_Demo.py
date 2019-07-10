import gym
from DQAgent import DQAgent

env = gym.make('CartPole-v0')

agent_opts = {
                'BATCH_SIZE':             16,
                'AGGREGATE_STATS_EVERY':   5, # not implemented
                'SHOW_EVERY':              5,
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .95,
                'DISCOUNT':              .90,
                'MAX_STEPS':             500,
                'MIN_EPSILON' :          .10,
                'REPLAY_MEMORY_SIZE':    500,
                'LEARNING_RATE':         0.01,
                'ACTION_POLICY':         'eg'
            } 

model_opts = {
                'num_layers':      2,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'add_dropout':     False,
                'activation':      'linear',
                'nodes_per_layer': [10, 10],
            }

agent = DQAgent(env, **agent_opts)
agent.build_model(**model_opts)
agent.load_weights('cartpole.h5')
#agent.train(n_epochs=200)
agent.evaluate(5)
env.close()
#agent.save_weights('cartpole')
print('Best Reward:', agent.best_reward)