import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#### SOMEHOW FIXES ISSUE WITH TENSORFLOW ####
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#### --- ####

# Tutorial: https://www.tensorflow.org/tutorials/keras/classification

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

# Examine picture at index 0: Notice that pixelvalues are in [0, 255] -- but we want [0, 1].
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Scale data to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# To see if it works:
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

# Now we are ready to build our neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flattens 2D Array to 1D Array
    tf.keras.layers.Dense(128, activation='relu'),      # First layer with 128 nodes
    tf.keras.layers.Dense(10)                           # Second (and last) layer with 10 nodes (because #labels = 10)
])

# Properties of our Model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train our model with test data
model.fit(train_images, train_labels, epochs=10)

# Compare Accuracy for Tests
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# Test Accuracy = 0.8753 < Trainings Accuracy = 0.9115  ==> Overfitting!

# Make Predictions:
# Softmax to convert protocols in probability
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Print out prediction for first test image
print(predictions[0]) # returns array with probability for each label
print(np.argmax(predictions[0])) # returns integer label i.e. label with highest probability


# Graphical Interpretation:
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))                                                 # x-Achse im Bereich [0, 9]
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Take image[0] and test model:
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()