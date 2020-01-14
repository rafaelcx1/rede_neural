import numpy as np
from network import Network

inputs = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1]
])

outputs = np.array([
  [0],
  [1],
  [1],
  [0]
])

network = Network.create([2,3,1])
network.print()
network.train(inputs, outputs, 0.1, 50000)
print(network.feed_forward(inputs[0]))