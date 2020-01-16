import pickle
import numpy as np
import matplotlib.pyplot as plt
from network import Network

def make_output(n):
  result = np.zeros((1, 10))
  result.itemset(int(n), 1.0)
  return result

def get_outputs(labels):
  result = np.array(make_output(labels.item(0)))
  labels = np.delete(labels, (0), axis=0)
  
  for label in labels:
    result = np.append(result, make_output(label), axis=0)

  return result

numbers_labels = np.array([
    list(
        open('data\\train-labels.idx1-ubyte', 'rb').read()[8:]
    )
]).T

numbers = np.array([
    list(
        open('data\\train-images.idx3-ubyte', 'rb').read()[16:]
    )
]).T

image_size = 28 * 28
qtd_images = int(len(numbers) / image_size)

inputs = numbers.reshape((qtd_images, image_size)) / 255
#outputs = get_outputs(numbers_labels)

#print(inputs.shape)
#print(outputs.shape)

#network = Network.create([784, 256, 128, 10])
#losses = network.train(inputs[0:10000], outputs[0:10000], 0.1, 15)

# with open('net.ser', 'wb') as file:
#   pickle.dump(network, file)

# with open('last_losses.ser', 'wb') as file:
#   pickle.dump(losses, file)

with open('net.ser', 'rb') as file:
  network = pickle.load(file)

with open('last_losses.ser', 'rb') as file:
  losses = pickle.load(file)

img_to_predict = 45999

plt.suptitle('Imagem Inserida')
plt.imshow((np.array([inputs[img_to_predict]])).reshape((28,28)), 'binary')
plt.show()

plt.subplot(121)
plt.bar(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], network.predict(inputs[img_to_predict])[0])
plt.subplot(122)
plt.imshow((np.array([inputs[img_to_predict]])).reshape((28,28)), 'binary')
plt.suptitle('Resultado')
plt.show()