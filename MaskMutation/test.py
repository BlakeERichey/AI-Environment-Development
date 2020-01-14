from neural import Network
from worker import Worker
import numpy as np

inputs = [
  [1,2,3,4],
  [2,3,4,5]
]
inputs = np.array(inputs)

net = Network()
net.add_layer(4, "input", None, False)
net.add_layer(256, "linear", None, False)
net.add_layer(256, "linear", None, False)
net.add_layer(2, "linear", None, False)
net.compile()

net2 = Network()
net2.add_layer(4, "input", None, False)
net2.add_layer(6, "linear", None, False)
net2.add_layer(6, "linear", None, False)
net2.add_layer(2, "linear", None, False)
net2.compile()

agent = Worker(net)
agent2 = Worker(net2)

print("Net1:", net.feed_forward(inputs))
print("Net2:", net2.feed_forward(inputs))

# new_weights = agent.breed(agent2)

# for i, weights in enumerate(new_weights):
#   agent.net.layers[i].weights = weights
agent.mutate()
print("New:", agent.net.feed_forward(inputs))