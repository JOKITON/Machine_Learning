""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

import config
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
import numpy as np
from df_utils import conv_binary, check_df_errors, normalize_df
from activation import sigmoid, relu, cross_entropy
from calc_loss import compute_mse, compute_mae, compute_r2score, compute_loss

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

LEARNING_RATE = 1
CONVERGENCE_THRESHOLD = 1e-10
EPOCHS = 2000  # Number of epochs
N_NEURONS = 0  # Number of neurons per layer 16->8->4->1
N_LAYERS = 0  # Number of layers

# Load the dataset
df = pd.read_csv(config.MULTILAYER_PERCEPTRON_CLEAN_DATASET)
# Drop unnecessary columns
df.drop(columns=['id'], inplace=True)

# Extract & Normalize the diagnosis column
y = df.pop('diagnosis')
y = conv_binary(y)

# Handle errors, unnamed columns, and NaN values
check_df_errors(df, False)
# Normalize the feature columns
X = normalize_df(df, "z-score")

# Split into training and test sets
print(Fore.GREEN + Style.DIM
    + "Splitting the dataset into training and test sets... (80/20)" + RESET_ALL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

""" 
# Check shapes for train_test_split

print("Shape of X_train & X_test: ", X_train.shape, X_test.shape)
print("Shape of Y_train & Y_test: ",y_train.shape, y_test.shape)
print()
"""

# Create a 2D matrix of weights for the neural network along with a bias vector
weights = np.random.rand(X_train.shape[1])
biases = 0

def stepforward(_X, _weights, _bias):
    """ 
        Apply forward propagation to get predictions.
        
        Parameters:
            - _X: Feature matrix, shape (n_samples, n_features)
            - _weights: Weights, shape (n_features,)
            - _bias: Bias, shape (1,)
            
        Returns:
            - predictions: Predicted target values, shape (n_samples,)
    """
    z = np.dot(_X, _weights) + _bias
    a = sigmoid(z)  # Apply sigmoid activation
    return a

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))  # Sigmoid derivative

def compute_gradient(_X, _y_train, _predicted, _weights, _biases):
    """
    Computes the gradient of the weights and biases.
    Update values accordingly using the given learning rate.
    
    Parameters:
        - _X: Feature matrix, shape (n_samples, n_features)
        - _y_train: Target values, shape (n_samples,)
        - _predicted: Predicted target values, shape (n_samples,)
        - _weights: Current weights, shape (n_features,)
        - _biases: Current biases, shape (1,)

    Returns:
        - new_weights: Updated weights, shape (n_features,)
        - new_biases: Updated biases, shape (1,)
    """
    n_samples = len(_predicted)

    # Derivative of MSE loss with respect to weights and bias
    dW = (2/n_samples) * np.dot(_X.T, (_predicted - _y_train))  # Gradient for weights
    db = (2/n_samples) * np.sum(_predicted - _y_train)  # Gradient for bias

    # Update weights and biases
    new_weights = _weights - LEARNING_RATE * dW.flatten()
    new_biases = _biases - LEARNING_RATE * db

    return new_weights, new_biases

# Example usage:
y_train = y_train.to_numpy()

prev_mse = None
for i in range(EPOCHS):
    W = np.asarray(weights).flatten()
    B = biases
    preds = stepforward(X_train, W, B)
    preds = relu(preds)
    
    # MSE
    mse = cross_entropy(y_train, preds)
    if prev_mse is not None and abs(mse - prev_mse) < CONVERGENCE_THRESHOLD:
        break
    prev_mse = mse

    weights, biases = compute_gradient(X_train, y_train, preds, W, B)
    if (i % 100 == 0):
        print(f"Training neuron {0}... Epoch {i}", f"MSE: {mse}")

# MSE, MAE, r2 score
mse = compute_mse(y_train, preds)
mae = compute_mae(y_train, preds)
r2 = compute_r2score(y_train, preds)
# print("Weights: ", weights)
print(f"(End of training) MSE for neuron {0}: {mse}")
print(f"(End of training) MAE for neuron {0}: {mae}")
print(f"(End of training) R2 Score for neuron {0}: {r2}")
# print(f"(End of training) Cross entropy {layer}: {cross_entropy(y_train, predictions)}")
print()

W = np.asarray(weights).flatten()
B = biases
predictions = stepforward(X_test, W, B)

test_mse = compute_mse(y_test, predictions)
test_mae = compute_mae(y_test, predictions)
test_r2 = compute_r2score(y_test, predictions)
print(f"Test MSE for neuron {0}: {test_mse}")
print(f"Test MAE for neuron {0}: {test_mae}")
print(f"Test R2 Score for neuron {0}: {test_r2}")
print()
