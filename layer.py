import numpy as np
from numpy import ndarray
from operations import sigmoid

class Layer:
  def __init__(self, neurons_qtd: int, last_neuron_layer_qtd: int, initial_bias: int = 0.1):
    self.weights = np.random.random((neurons_qtd, last_neuron_layer_qtd)) - 0.5
    self.bias = np.full((1, neurons_qtd), initial_bias)
    self.sum_cache = np.array([])
    self.output_cache = np.array([])
    self.error_derivative = np.zeros(self.weights.shape)
    self.output_derivative = np.zeros((1, neurons_qtd))
    self.input_derivative = np.zeros((1, neurons_qtd))
  
  def feed_forward(self, input: ndarray) -> ndarray:
    self.sum_cache = np.dot(self.weights, input.T).T + self.bias
    self.output_cache = sigmoid(self.sum_cache)
    return self.output_cache

  def clean_derivatives(self) -> None:
    self.error_derivative = np.zeros(self.weights.shape)
    self.output_derivative = np.zeros(self.output_derivative.shape)
    self.input_derivative = np.zeros(self.input_derivative.shape)

  def print(self) -> None:
    text = (
        'weights: ' + str(self.weights),
        'bias: ' + str(self.bias),
        'sum_cache: ' + str(self.sum_cache),
        'output_cache: ' + str(self.output_cache),
        'error_derivative: ' + str(self.error_derivative),
        'output_derivative: ' + str(self.output_derivative),
        'input_derivative: ' + str(self.input_derivative)
    )
    print(str(text) + '\n')
