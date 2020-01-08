import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)
  
def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()
