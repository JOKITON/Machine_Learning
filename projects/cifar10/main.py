""" CIFAR-10 image classification using a Convolutional Neural Network """

import os
import sys
import numpy as np
from tensorflow import keras
from keras import models, layers

# Add the directory containing config.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

NB_CLASSES = 10
NUM_EPOCHS = 20

# Load the CIFAR-10 dataset
if not os.path.exists(config.CIFAR10_PATH):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Save the dataset locally
    np.savez_compressed(config.CIFAR10_PATH,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
else:
    data = np.load(config.CIFAR10_PATH)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

# Normalize pixel numbers to (0, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Normalize categorical data increasing the number of columns
y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=NUM_EPOCHS, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
