""" Activation functions for neural networks. """

import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ ReLU activation function. """
    return np.maximum(0, x)

def cross_entropy(y_true, y_pred):
    """ Cross Entropy activation function. """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
