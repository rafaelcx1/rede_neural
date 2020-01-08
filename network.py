import numpy as np
from neuron import Neuron
from operations import deriv_sigmoid

class Network:
    def __init__(self, layersDimensions):
        self.dimensions = layersDimensions
        self.outputs = []
        self.layers = []
        self.initializeLayers(layersDimensions)
        
    def initializeLayers(self, layersDimensions):
        for layerIndex in range(1, len(layersDimensions)):
            layer = []
            
            for indexNeuron in range(1, layersDimensions[layerIndex] + 1):
                lastLayerDimension = layersDimensions[layerIndex - 1]
                actualLayerDimension = layersDimensions[layerIndex]

                weights = np.random.random(lastLayerDimension)
                bias = np.random.random()
                neuron = Neuron(weights, bias)
                
                layer.append(neuron)
                
            self.layers.append(layer)
    
    def showLayers(self):
        l = 1
        for layer in self.layers:
            print('layer: ' + str(l))
            i = 1
            l += 1
            for neuron in layer:
                print('neuron ' + str(i) + ': ')
                print(neuron.weights)
                print(neuron.bias)
                print('---')
                i += 1
    
    def feedForward(self, inputs):
        nextInput = inputs
        
        for layer in self.layers:
            outputs = []
            
            for neuron in layer:
                outputs.append(neuron.output(nextInput))
            
            nextInput = outputs
            self.outputs.append(outputs)
            
        return outputs
    
    def backpropLayer(self, actual_layer, last_layer):
      for neuron in actual_layer:
        for weight, neuronLastLayer in zip(neuron.weights, last_layer):
          pass

    def train(self, inputs, correct_outputs, epochs, learn_rate):
      for epoch in range(epochs):
        for inputs, correctOutput in zip(inputs, correct_outputs):
          network = Network(self.dimensions)
          outputResult = network.feedForward(inputs)

          loss = -2 * (correctOutput - outputResult)

          for layerIndex in range(len(network.layers) - 1, 1):
            backpropLayer(network.layers[layerIndex], network.layers[layerIndex - 1])



