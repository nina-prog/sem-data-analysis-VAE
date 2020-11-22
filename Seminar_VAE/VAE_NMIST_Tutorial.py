import numpy as np
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

##########################################################################################
# IN THE FOLLOWING WE DO THE PAPERSPACE VAE TUTORIAL ON MNIST NUMBER DATASET #
# (Use Juypter Notebook for step to step learning) #
# Link: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
##########################################################################################
# Code can be splitted into the following sections:
#   (1) Create Encoder
#   (2) Create Decoder
#   (3) Building the VAE
#   (4) Train the VAE
#   (5) Test the VAE
##########################################################################################

######################
### Create Encoder ###
######################
# TASK: Encoder gets Input Image with Shape 28x28 and returns latent vector of length 2 (mean and std)

# Image Properties
img_size = 28
num_channels = 1

# Input Layer
x = tf.keras.layers.Input(shape=(img_size, img_size, num_channels), name="encoder_input")

encoder_conv_layer1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="encoder_conv_1")(x)
encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
encoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)

encoder_conv_layer2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=1, name="encoder_conv_2")(encoder_activ_layer1)
encoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
encoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)

encoder_conv_layer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2, name="encoder_conv_3")(encoder_activ_layer2)
encoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
encoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

encoder_conv_layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2, name="encoder_conv_4")(encoder_activ_layer3)
encoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
encoder_activ_layer4 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_4")(encoder_norm_layer4)

encoder_conv_layer5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=1, name="encoder_conv_5")(encoder_activ_layer4)
encoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
encoder_activ_layer5 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_5")(encoder_norm_layer5)

shape_before_flatten = tf.keras.backend.int_shape(encoder_activ_layer5)[1:]
encoder_flatten = tf.keras.layers.Flatten()(encoder_activ_layer5)

latent_space_dim = 2

encoder_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
encoder_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)

encoder_mu_log_variance_model = tf.keras.models.Model(x, (encoder_mu, encoder_log_variance), name="encoder_mu_log_variance_model")

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

encoder = tf.keras.models.Model(x, encoder_output, name="encoder_model")

# To print out layer properties, use encoder.summary()
encoder.summary()


######################
### Create Decoder ###
######################
# TASK: Should do the opposite: Get latent vector of size 2 and return a 28x28 px Image

# First layer: Note the input with the shape of the latent_space_dim i.e. 2
decoder_input = tf.keras.layers.Input(shape=(latent_space_dim), name="decoder_input")

# Dense layer that expands the length of the vector from 2 to the value specified in shape_before_flatten variable
# which is (7, 7, 64). Now the Shape of the dense layer is (None, 3136)
decoder_dense_layer1 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten), name="decoder_dense_1")(decoder_input)

# Reshape the result of a vector back to a matrix using reshape layer
decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)

# After that, we can add a number of layers that expand the shape
# until reaching the desired shape of the original input, (28, 28).
decoder_conv_tran_layer1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_1")(decoder_reshape)
decoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
decoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

decoder_conv_tran_layer2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_2")(decoder_activ_layer1)
decoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
decoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

decoder_conv_tran_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_3")(decoder_activ_layer2)
decoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
decoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

decoder_conv_tran_layer4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_4")(decoder_activ_layer3)
decoder_output = tf.keras.layers.LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4 )

# Architecture of the decoder is done.
# Now create a model that links the input and output of the decoder
decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")

decoder.summary()


########################
### Building the VAE ###
########################
# TASK: We now create a VAE that combines the Encoder and the Decoder

# Input layer of the VAE
vae_input = tf.keras.layers.Input(shape=(img_size, img_size, num_channels), name="VAE_input")

# Connect VAE input layer to the encoder to encode the input and return the latent vector
vae_encoder_output = encoder(vae_input)

# The output of the encoder is then connected to the decoder to reconstruct the input.
vae_decoder_output = decoder(vae_encoder_output)

# Finally, the model of the VAE that links the encoder to the decoder is created.
vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")

vae.summary()

# Loss Function Implementation
def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

# Compile VAE
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=loss_func(encoder_mu, encoder_log_variance))


########################
### Training the VAE ###
########################

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Resize from (28, 28) to (28, 28, 1) (model expects the latter one)
x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))

# Train the VAE
vae.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

# Safe models for later use
encoder.save("VAE_encoder.h5")
decoder.save("VAE_decoder.h5")
vae.save("VAE.h5")


########################
### Testing the VAE ####
########################

# Load models from (5) Training the VAE
encoder = tf.keras.models.load_model("VAE_encoder.h5")
decoder = tf.keras.models.load_model("VAE_decoder.h5")

# Make sure data is loaded, too
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))

# Encode and decode the test data
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

