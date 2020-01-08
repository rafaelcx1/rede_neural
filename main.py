import numpy as np
from network import Network
from neuron import Neuron

n = Network([2,2,1])
n.feedForward([-2, -1])
print(n.outputs)