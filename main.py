import numpy as np
from network import Network

inputs = np.array([
  [0,0,0,0],
  [0,1,0,0],
  [1,0,0,0],
  [1,1,0,0]
])

outputs = np.array([
  [0],
  [0],
  [1],
  [1]
])

network = Network.create([4,4,4,1])

network.train(inputs, outputs, 1, 0.1)
print(network.predict(np.array([1, 0, 0, 0])))