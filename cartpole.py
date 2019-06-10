#A demo of Tensorflow's tf_agents
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
# import pyvirtualdisplay

import tensorflow as tf

from tf_agents.utils          import common
from tf_agents.agents.dqn     import dqn_agent
from tf_agents.environments   import suite_gym
from tf_agents.networks       import q_network
from tf_agents.metrics        import tf_metrics
from tf_agents.trajectories   import trajectory
from tf_agents.eval           import metric_utils
from tf_agents.policies       import random_tf_policy
from tf_agents.environments   import tf_py_environment
from tf_agents.drivers        import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

tf.compat.v1.enable_v2_behavior()


# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

env_name = 'CartPole-v0'         # @param
num_iterations = 20000           # @param

initial_collect_steps = 1000     # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 100000  # @param

fc_layer_params = (100,)

batch_size = 64                  # @param
learning_rate = 1e-3             # @param
log_interval = 200               # @param

num_eval_episodes = 10           # @param
eval_interval = 1000             # @param

env = suite_gym.load(env_name)

#@test {"skip": true}
env.reset()
env.render()