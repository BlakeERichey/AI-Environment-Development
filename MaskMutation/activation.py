import numpy as np

class Activation:

  def __init__(self, method):
    assert method in ["sigmoid", "tanh", "linear", "relu", "softmax"], "Invalid Activation function"

    self.function = method
  
  def activate(self, arr):
    assert isinstance(arr, np.ndarray)

    if self.function == "sigmoid":
      return 1/(1 + np.exp(-arr))
    if self.function == "tanh":
      return np.tanh(arr)
    if self.function == "relu":
      return np.maximum(arr, 0)
    if self.function == "softmax":
      if arr.ndim == 1:
        arr = arr.reshape((1, -1))
      max_arr = np.max(arr, axis=1).reshape((-1, 1))
      exp_arr = np.exp(arr - max_arr)
      return exp_arr / np.sum(exp_arr, axis=1).reshape((-1, 1))
    
    return arr
  