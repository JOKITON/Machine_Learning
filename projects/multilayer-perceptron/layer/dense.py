""" File to create the DenseLayer class. """

import numpy as np
from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, softmax

class DenseLayer:
    """ DenseLayer class used for creating a dense layer in a neural network. """
    # General Use
    out_layer = False
    # Forward Propagation
    inputs = None
    weights = None
    biases = None
    output = None
    activations = None
    # Backwward Propagation
    dinputs = None

    def __init__(self, n_inputs, n_neurons, activation_function, der_actv, f_dloss_actv):
        """
        Initialize the Dense layer.
        
        Parameters:
        n_inputs  - Number of inputs to the layer.
        n_neurons - Number of neurons in the layer.
        """
        # Weight initialization
        if n_neurons == 1:  # Output layer (Sigmoid)
            scale = np.sqrt(1 / n_inputs)
        else:  # Hidden layers (Leaky ReLU)
            scale = np.sqrt(2 / n_inputs)
        self.weights = np.random.randn(n_inputs, n_neurons) * scale
        self.biases = np.zeros((1, n_neurons))

        self.f_actv = activation_function
        self.f_der_actv = der_actv
        self.f_dloss_actv = f_dloss_actv

    def forward(self, inputs=None):
        """ 
		Perform forward propagation for the layer
  
		Parameters:
		inputs	- Input data

		Returns:
		Output of the layer after forward propagation.
        """
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

        self.activations = self.f_actv(self.output)

        return self.activations, self.output

    def backward(self, yvalues, learning_rate):
        """ 
		Perform backward propagation for the layer
  
		Parameters:
		dvalues			- Gradient of the loss with respect to the output.
		learning_rate	- The rate for gradient descent
        """
        if (self.f_dloss_actv == "cross_entropy"): # Cross Entropy Partial derivative
            dactivation = self.activations - yvalues
        elif (self.f_dloss_actv == "mse"): # MSE Partial derivative
            der_loss_activations = 2 * (self.activations - yvalues)
            der_activation_zvalues = self.f_der_actv(self.output)
            dactivation = (der_loss_activations * der_activation_zvalues)

        # Gradients with respect to weights, biases, and inputs
        dweights = np.dot(self.inputs.T, dactivation) / self.activations.shape[0]
        dbiases = np.sum(dactivation, axis=0, keepdims=True) / self.activations.shape[0]

        self.dinputs = np.dot(dactivation, self.weights.T)

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases
