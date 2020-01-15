import numpy as np
from operations import deriv_sigmoid
from operations import mse_loss
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
      print('Layer: ' + str(index + 1), ' | Dimension: ' + str(self.dimensions[index + 1]))
      self.layers[index].print()

  def feed_forward(self, input: ndarray) -> ndarray:
    self.input = input
    
    for layer in self.layers:
      input = layer.feed_forward(input)
    
    return input

  def predict(self, input) -> ndarray:
    return self.feed_forward(input)

  def train(self, inputs, outputs, learning_rate, epochs, loss_max = None) -> list:
    losses = []
    for epoch in range(epochs):
      print('Epoch: ' + str(epoch + 1))
      loss = 0
      
      for input, output in zip(inputs, outputs):
        output_predict = self.feed_forward(input)
        loss += mse_loss(output, output_predict)
        
        self.backprop(input, output)
        self.update_weights(learning_rate)

      loss /= len(outputs)
      losses.append(loss)
      print('Loss: ' + str(loss), end='\n\n')

      if loss_max != None and loss < loss_max:
        break

    return losses

  def backprop(self, input, output) -> None:

    qtd_layers = len(self.layers)

    last_layer = self.layers[qtd_layers - 1]
    last_layer.output_derivative = last_layer.output_cache - output

    for layer_index in range(qtd_layers - 1, -1, -1):
      layer = self.layers[layer_index]
      
      input_derivative = np.multiply(deriv_sigmoid(layer.sum_cache), layer.output_derivative)
      layer.input_derivative = layer.input_derivative + input_derivative

      last_output = None

      if layer_index == 0:
        last_output = input
      else:
        last_layer = self.layers[layer_index - 1]
        last_layer.output_derivative = last_layer.output_derivative + np.multiply(input_derivative, layer.weights.T).sum(axis=1)
        last_output = last_layer.output_cache

      error_derivative = input_derivative.T * last_output
      layer.error_derivative = layer.error_derivative + error_derivative

  def update_weights(self, learning_rate) -> None:
    for layer in self.layers:
        layer.weights = layer.weights - learning_rate * layer.error_derivative
        layer.bias = np.reshape(layer.bias - learning_rate * layer.input_derivative, layer.bias.shape)
        layer.clean_derivatives()
      
