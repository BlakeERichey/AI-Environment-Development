import numpy as np
import math
from activation import Activation

class Layer:

  def __init__(self, rows, cols, activation='linear', weights=None, use_bias=True):
    self.rows = rows
    self.cols = cols
    self.activatation = Activation(activation)

    if isinstance(weights, np.ndarray):
      self.weights = weights
    else:
      limit = math.sqrt(6/(rows+cols)) #glorot uniform
      self.weights = np.random.uniform(low=-limit, high=limit, size=(rows*cols,)).reshape((rows, cols))
    
    if use_bias:
      self.bias = np.random.uniform(low=-1, high=1, size=(cols,))
    else:
      self.bias = None
  
  def set_weights(self, weights):
    assert isinstance(weights, np.ndarray), f"Invalid weights type. Wanted NumPy array, got {type(weights)}."

    self.weights = weights
  
  def set_bias(self, bias):
    assert isinstance(bias, np.ndarray), f"Invalid bias type. Wanted NumPy array, got {type(weights)}."

    self.bias = bias
  
  def __str__(self,):
    return f'{self.rows} {self.cols} \n{self.weights}'