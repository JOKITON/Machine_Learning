""" File to create the DenseLayer class. """

import numpy as np

class DenseLayer:
    """ DenseLayer class used for creating a dense layer in a neural network. """
    inputs = None
    output = None
    dweights = None
    dbiases = None
    dinputs = None

    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the Dense layer.
        
        Parameters:
        n_inputs  - Number of inputs to the layer.
        n_neurons - Number of neurons in the layer.
        """
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        """ 
		Perform forward propagation for the layer
  
		Parameters:
		inputs	- Input data

		Returns:
		Output of the layer after forward propagation.
        """
        self.inputs = inputs
        self.output = (self.inputs * self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        """ 
		Perform backward propagation for the layer
  
		Parameters:
		dvalues			- Gradient of the loss with respect to the output.
		learning_rate	- The rate for gradient descent
        """

        # Gradients with respect to weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
