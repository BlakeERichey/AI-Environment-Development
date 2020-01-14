from layer import Layer
import numpy as np

class Network:

  def __init__(self):
    self.layers = None
    self.input = None
    self.output = None
  
  def add_layer(self, nodes, activation="", weights=None, use_bias=True):
    assert nodes > 0 and type(nodes) == int, "Invalid quantity of nodes"

    rows = 0
    if self.input is None:
      assert activation in ["input", ""], "No activation should be provided for input layer."
      self.input = nodes
    elif self.layers is None:
      rows = self.input
    else:
      rows = self.layers[-1].cols

    if rows > 0:
      if self.layers is None:
        self.layers = []
      self.layers.append(Layer(rows, nodes, activation, weights, use_bias))
  
  def compile(self):
    self.output = np.size(self.layers[-1].weights, 1) #number of cols in last layer

  def feed_forward(self, data):
    assert self.output is not None, "Uncompiled model or no output nodes found"
    assert isinstance(data, np.ndarray), "Invalid data type for inputs"

    hi = data #hidden layer input
    for layer in self.layers:
      ho = np.dot(hi, layer.weights)
      if layer.bias:
        ho = np.add(ho, layer.bias)
      ho = layer.activatation.activate(ho)
      hi = ho

    return hi

  def clone(self):
    net = Network()
    net.input = self.input
    net.output = self.output
    net.layers = []
    for layer in self.layers:
      net.layers.append(layer.clone())
    
    return net
  
  def __str__(self,):
    string = ''
    for layer in self.layers:
      string+=str(layer)+'\n'
    return string