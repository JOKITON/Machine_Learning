"" "Trains a simple deep NN on the MNIST dataset. " ""

# import tensorflow as tf
# import numpy as np
from tensorflow import keras

# Define network & training params
EPOCHS = 300
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION

# Load MNIST dataset
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# We are gonna train with 60.000 samples and test with 10.000 samples

# X_train is 60000 rows of 28x28 values; we --> reshape it to 60000 x 784.
RESHAPED = 784

# Reshape dataset to use bits in each space
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize inputs to be within in [0, 1].
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot representation of the labels.
Y_train = keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model.
model = keras.models.Sequential()
model.add(keras.layers.Dense(
    NB_CLASSES,
    input_shape=(RESHAPED,),
    name='dense_layer',
    activation='softmax')
)
