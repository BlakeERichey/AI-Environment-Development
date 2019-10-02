import gym, os
from agent import DQAgent

env = gym.make('BattleZone-v0')

#for epoch in range(1):
#  num_steps = 0
#  done      = False
#  envstate  = env.reset()
#  while not done and num_steps < 500: #perform action/step
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#    env.render()
#    num_steps += 1

root_path = './'

agent_opts = {
                #hyperparameters
                'REPLAY_BATCH_SIZE':      16,
                'LEARNING_BATCH_SIZE':     6,
                'DISCOUNT':              .9,
                'MAX_STEPS':             1000,
                'REPLAY_MEMORY_SIZE':    500,
                'LEARNING_RATE':         0.001,
                
                #ann specific
                'EPSILON_START':         .90,
                'EPSILON_DECAY':         .99,
                'MIN_EPSILON' :          0.01,

                #saving and logging results
                'AGGREGATE_STATS_EVERY':  5,
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
    agent.load_weights(f'{root_path}best_model')
    agent.train(n_epochs=1000, render=False)
    agent.save_weights(f'{root_path}racing2')
    agent.show_plots('cumulative')
    agent.show_plots('loss')
    env.close()

#Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    if best_model:
      filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
      agent.load_weights(filename)
    else:
      agent.load_weights(f'{root_path}racing2')
    results = agent.evaluate(100, render=True, verbose=True)
    print(f'Average Results: {sum(sum(results,[]))/len(results)}')

#train_model(agent_opts, model_opts)
evaluate_model(agent_opts, model_opts, best_model=True)