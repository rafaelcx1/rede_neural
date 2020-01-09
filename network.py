import numpy as np
from operations import deriv_sigmoid
from numpy import ndarray
from layer import Layer

class Network:
  def __init__(self):
    self.input = None
    self.dimensions = None
    self.layers = None
  
  @staticmethod
  def create(dimensions: []):
    network = Network()
    network.dimensions = dimensions
    network.layers = []

    for dimension_index in range(1, len(dimensions)):
      last_dimension = dimensions[dimension_index - 1]
      actual_dimension = dimensions[dimension_index]

      network.layers.append(Layer(actual_dimension, last_dimension))
    
    return network

  def print(self) -> None:
    print('Input Layer | Dimension: ' + str(self.dimensions[0]) + '\n')

    for index in range(0, len(self.layers)):
      print('Layer: ' + str(index + 1), ' | Dimension: ' + str(self.dimensions[index] + 1))
      self.layers[index].print()

  def feed_forward(self, input: ndarray) -> ndarray:
    self.input = input
    
    for layer in self.layers:
      input = layer.feed_forward(input)
    
    return input

  def train(self, inputs, outputs, epochs, learning_rate) -> None:
    for epoch in range(1, epochs + 1):
      for input, output in zip(inputs, outputs):
        predicts = self.feed_forward(input)
        weights_gradients = self.get_grads(input, output, predicts, learning_rate)
        
        for w_grad, layer in zip(weights_gradients, self.layers):
          layer.weights -= w_grad

  def predict(self, input) -> ndarray:
    return self.feed_forward(input)

  def get_grads(self, input, output, predicts, learning_rate):
    layers = self.layers
    d_loss = 2 * (output - predicts)
    d_predicts = deriv_sigmoid(predicts)

    d_weights = []
    
    for layer_index in range(len(layers) - 1, -1, -1):
      is_output_layer: bool = layer_index == len(layers) - 1
      is_input_layer: bool = layer_index == 0

      last_layer_outputs = None
      next_layer_weights = None
      middle_formula = None

      if is_input_layer:
        last_layer_outputs = np.array([input])
      else:
        last_layer_outputs = layers[layer_index - 1].output_cache

      if is_output_layer:
        middle_formula = d_loss * d_predicts
      else:
        next_layer_weights = layers[layer_index + 1].weights
        middle_formula = np.dot(d_loss * d_predicts, next_layer_weights) * np.array([deriv_sigmoid(layers[layer_index].output_cache)])

      d_weights.append(learning_rate * np.dot(np.array([last_layer_outputs]).T, np.array([middle_formula])))
    
    return d_weights

