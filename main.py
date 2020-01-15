import numpy as np
import matplotlib.pyplot as plt
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
losses = network.train(inputs, outputs, 0.1, 100000)

plt.plot(losses);
plt.show()