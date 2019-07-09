import gym
from DQAgent import DQAgent

env = gym.make('CartPole-v0')

agent = DQAgent(env)
agent.build_model(add_dropout=True)

# envstate, action, reward, next_envstate, done

for _ in range(5): #epoch
 obs = env.reset()
 done = False
 for _ in range(10): #steps
   prev_state = obs
   action = env.action_space.sample()
   obs, reward, done, _ = env.step(action)

   episode = [prev_state, action, reward, obs, done]
   agent.remember(episode)
   env.render()

   if done:
     print('Result:', reward)
     break    