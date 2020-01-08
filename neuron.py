import numpy as np
from operations import sigmoid

class Neuron:
  def __init__(self, weights, bias):
    self.weights = np.array([1,1]) #weights
    self.bias = 0 #bias
    self.outputCalculated = None
    self.sum = None

  def output(self, inputs):
    if self.outputCalculated is None:
      total = np.dot(self.weights, inputs) + self.bias
      self.sum = total
      self.outputCalculated = sigmoid(self.sum)
    
    return self.outputCalculated