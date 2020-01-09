from numpy import ndarray
from layer import Layer

class Network:
  def __init__(self):
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
    for index in range(0, len(self.layers)):
      print('Layer: ' + str(index + 1))
      self.layers[index].print()

  def feed_forward(self, input: ndarray) -> ndarray:
    for layer in self.layers:
      input = layer.feed_forward(input)
    
    return input
