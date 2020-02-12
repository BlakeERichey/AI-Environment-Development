from model import ActorCritic
import gym
import numpy as np 

def make_env(params, rank=0):
  env = gym.make(params.get('env_name'))
  env.seed(params.get('seed', 0) + rank)
  return env

def train(rank, params, critic, n_timesteps=4):
  '''
    creates an actor runs it through environment.
    Passes learned information back to shared critic

    :rank is the actor number
    :params is a dictionary with relevant environment information in order to construct it
  '''
  
  #create env and seed
  env = make_env(params, rank)
  
  #create new actor
  agents = ActorCritic(env, n_timesteps)
  agents.actor = agents.create_actor()
  agents.critic = critic #use shared critic, for feed_forward function

  #run through environment
  ob = env.reset()
  ob = np.array([ob for _ in range(n_timesteps)])
  done = False
  num_steps = 0
  max_steps = params.get('num_steps', None)
  while not done and (True,num_steps<max_steps)[max_steps is not None]:
    action, state_value = agents.feed_forward(ob) #get actor,critic response
    print(action, state_value)
    # action = env.action_space.sample()
    envstate, reward, done, _ = env.step(action) 

    #add to timestep oberservation, remove oldest envstate from ob
    ob = np.concatenate((ob, np.expand_dims(envstate, axis=0)), axis=0)[1:]
    num_steps+=1
    env.render()
  
  env.close()