from typing import Optional, Tuple, Union
from image_processor import ImageProcessor
import network
import image_loader
import mnist_loader
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# training_data = list(training_data)
# ip = ImageProcessor((28, 28))

# modified_training_data = []
# for i in range(len(training_data)):
#   pixels = training_data[i][0]
#   pixels = np.reshape(pixels, (28, 28))
#   pixels = ip.process(pixels)
#   pixels = np.reshape(pixels, (784, 1))

#   modified_training_data.append((pixels, training_data[i][1]))


epochs = 300
mbs = 100
lr = 3

folder = f"data/mnist{epochs}-lr{lr}-mbs{mbs}"

net = network.Network([784, 30, 10])
net.initialize_weights_and_biases()
net.SGD(training_data, epochs, mbs, lr, folder, test_data=test_data)
net.save_weights_and_biases(folder)


# il = image_loader.ImageLoader((28, 28), (280, 280), "images/grids")
# images = il.images

# print(images)

# for im in images:

#   output = net.feedforward(np.reshape(im[0], (784,1)))
#   highest = 0
#   i = 0
#   for idx, a in enumerate(output):
#       # print(str(idx) + " - " + "{:.3f}".format(a[0] * 100) + " %")
#       if a[0] > highest:
#           highest = a[0]
#           i = idx
#   print("\n")
#   print(f"Actual - {im[1]+1}")
#   print(str(i) + " - " + "{:.3f}".format(highest * 100) + " %")
