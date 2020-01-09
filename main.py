import numpy as np
from layer import Layer

input = np.array([1, 1])

layer: Layer = Layer(2, 2)

print(layer.feed_forward(input))

# for layer in layers:
#     input = np.dot(layer['weights'], input.T) + layer['bias']
#     print(type(layer['weights']))