import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load Data from tensorflow.keras
fashion_mnist = keras.datasets.fashion_mnist

# Pictures are 28x28 Numpy Arrays with pixel value in [0, 255], Labels are in [0, 9]
# Basically a 28x28 Matrix (for better understanding): Do "print(train_images[0])" to see
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for plotting later; basically an bijective function from labels to class_names
# i.e. label 0 = T-Shirt/top, label 1 = Trouser, etc...
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Each Set has 60.000 Pictures with each 28x28 pixel size
print(train_images.shape)
print(train_labels.shape)           # Label Size 60000, so each picture is mapped to a label

