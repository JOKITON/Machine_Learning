""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

import config
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import conv_binary, check_df_errors, normalize_df
from activations import sigmoid, relu, cross_entropy
from metrics import compute_mse, compute_mae, compute_r2score
from loss import compute_loss
from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD, EPOCHS, N_NEURONS, N_LAYERS, DIVIDE_NEURONS
from train import forward_propagation, compute_gradients, sigmoid_derivative

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

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
    X, y, test_size=0.25, random_state=42, stratify=y)

""" 
# Check shapes for train_test_split

print("Shape of X_train & X_test: ", X_train.shape, X_test.shape)
print("Shape of Y_train & Y_test: ",y_train.shape, y_test.shape)
print()
"""

def calc_neuron_shape(n_neurons, n_actual_layer, n_layers):
    layers_to_go = n_layers - n_actual_layer

def get_neurons(n_neurons, n_layers):
    weights = []
    biases = []
    prev_neurons = 0
    num_neurons = n_neurons
    for it in range(n_layers):
        if it == 0:
            weights.append(np.random.randn(X_train.shape[1], n_neurons) * np.sqrt(1 / X_train.shape[1]))
            biases.append(np.zeros((1, n_neurons)))
        elif it == n_layers - 1: # Output layer
            if prev_neurons == 0:
                prev_neurons = n_neurons
            weights.append(np.random.randn(prev_neurons, 1) * np.sqrt(1 / n_neurons))
            biases.append(np.zeros((1, 1)))
        else:
            prev_neurons = num_neurons
            if (DIVIDE_NEURONS):
                num_neurons = int(num_neurons / 2)
            print(num_neurons)
            print("Shape of hidden layer: (" + f"{prev_neurons}, {num_neurons})")
            weights.append(np.random.randn(prev_neurons, num_neurons) * np.sqrt(1 / n_neurons))
            biases.append(np.zeros((1, num_neurons)))
            prev_neurons = num_neurons

    return weights, biases

# Create weights and biases for all layers
weights, biases = get_neurons(N_NEURONS, N_LAYERS)

"""
#* Print data & shapes of weights/biases
for i in range(len(weights)):
    print("Weights shape [Layer "+ f"{i}]\n", weights[i])
for i in range(len(biases)):
    print("Biases shape: [Layer "+ f"{i}]\n", biases[i])
"""

y_train = y_train.to_numpy().reshape(-1, 1)

for epoch in range(EPOCHS):
    # Forward propagation
    if epoch % STEP_SIZE == 0:
        LEARNING_RATE *= DECAY_RATE
    activations, z_values = forward_propagation(X_train, weights, biases)
    
    # Compute loss (e.g., Mean Squared Error or Cross-Entropy)
    predictions = activations[-1]  # Output from the last layer
    loss = compute_loss(y_train, predictions)

    # Backward propagation (compute gradients and update weights/biases)
    resW, resB = compute_gradients(activations, z_values, y_train, weights, biases, sigmoid_derivative)  # Define this function
    
    # Update weights and biases for each layer
    for i in range(N_LAYERS):
        weights[i] -= LEARNING_RATE * resW[i]
        biases[i] -= LEARNING_RATE * resB[i]

    # Optional: Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

activations[-1] = np.round(activations[-1])
for i in range(len(activations[-1])):
    if (activations[-1][i] != y_train[i]):
        print(f"Incorrect prediction: {activations[-1][i]}")

# MSE, MAE, r2 score
mse = compute_mse(y_train, activations[-1])
mae = compute_mae(y_train, activations[-1])
r2 = compute_r2score(y_train, activations[-1])

print()
print(f"Training MSE: {mse}")
print(f"Training MAE: {mae}")
print(f"Training R2 Score: {r2}")
print()

y_test = y_test.to_numpy().reshape(-1, 1)
# Forward propagation
activations, z_values = forward_propagation(X_test, weights, biases)

# Compute loss (e.g., Mean Squared Error or Cross-Entropy)
predictions = activations[-1]  # Output from the last layer
loss = compute_loss(y_test, predictions)

mse = compute_mse(y_test, activations[-1])
mae = compute_mae(y_test, activations[-1])
r2 = compute_r2score(y_test, activations[-1])

print()
print(f"Testing MSE: {mse}")
print(f"Testing MAE: {mae}")
print(f"Testing R2 Score: {r2}")
print()
