import numpy as np
from numpy import ndarray
from operations import sigmoid

class Layer:
  def __init__(self, neurons_qtd: int, last_neuron_layer_qtd: int):
    self.weights = np.random.random((neurons_qtd, last_neuron_layer_qtd))
    self.bias = np.zeros(neurons_qtd)
    self.sum_cache = np.array([])
    self.output_cache = np.array([])
    self.error_derivative = np.zeros(self.weights.shape)
    self.output_derivative = np.zeros(self.weights.shape)
    self.input_derivative = np.zeros(neurons_qtd)
  
  def feed_forward(self, input: ndarray) -> ndarray:
    self.sum_cache = np.dot(self.weights, input.T) + self.bias
    self.output_cache = sigmoid(self.sum_cache)
    return self.output_cache

  def print(self) -> None:
    text = (
        'weights: ' + str(self.weights),
        'bias: ' + str(self.bias),
        'sum_cache: ' + str(self.sum_cache),
        'output_cache: ' + str(self.output_cache)
    )
    print(str(text) + '\n')
