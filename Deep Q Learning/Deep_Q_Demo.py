import gym
from DQAgent import DQAgent

env = gym.make('MountainCar-v0')

agent_opts = {
                'BATCH_SIZE':             16,
                'AGGREGATE_STATS_EVERY':   5, # not implemented
                'SHOW_EVERY':              2,
                'EPSILON_START':         .99,
                'EPSILON_DECAY':         .99,
                'DISCOUNT':              .90,
                'MAX_STEPS':             200,
                'MIN_EPSILON' :          .15,
                'REPLAY_MEMORY_SIZE':    500,
                'LEARNING_RATE':         0.05,
                'ACTION_POLICY':         'softmax'
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
agent.train(n_epochs=800)
agent.evaluate(5)
env.close()
print('Best Reward:', agent.best_reward)

# envstate, action, reward, next_envstate, done

# for _ in range(5): #epoch
#  obs = env.reset()
#  done = False
#  for _ in range(10): #steps
#    prev_state = obs
#    action = env.action_space.sample()
#    obs, reward, done, _ = env.step(action)

#    episode = [prev_state, action, reward, obs, done]
#    agent.remember(episode)
#    env.render()

#    if done:
#      print('Result:', reward)
#      break    