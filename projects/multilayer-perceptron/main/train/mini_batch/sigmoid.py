""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

import config
from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
from config import EPOCHS_MINI_BATCH_2, LS_SIGMOID_2, N_LAYERS, BATCH_SIZE
import pandas as pd
from colorama import Fore, Back, Style
import numpy as np
from batch import get_batches, shuffle_batches, get_val_batches
from preprocessing import get_train_test_pd
from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, tanh, der_tanh, softmax, der_softmax
from loss import f_loss, f_mse, f_mae, f_r2score, f_cross_entropy
from dense import DenseLayer
from plot import Plot
from plots import plot_acc_epochs, plot_loss_epochs

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL
EPOCHS = EPOCHS_MINI_BATCH_2
LAYER_SHAPE = LS_SIGMOID_2

# Normalize the data
X_train, y_train, X_test, y_test = get_train_test_pd()
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

layers = []
is_output = False
for layer in range(N_LAYERS):
    prev_neurons = LAYER_SHAPE[layer][0]
    out_neurons = LAYER_SHAPE[layer][1]
    if (layer == (N_LAYERS - 1)):
        f_actv = sigmoid
        f_der_actv = der_sigmoid
        f_dloss_actv = "cross_entropy"
    else:
        f_actv = leaky_relu
        f_der_actv = der_leaky_relu
        f_dloss_actv = "cross_entropy"
    # print(f"Layer {layer} - {prev_neurons} -> {out_neurons}")
    layers.append(DenseLayer(prev_neurons, out_neurons, f_actv, f_der_actv, f_dloss_actv))

"""
#* Print data & shapes of weights/biases
for i in range(len(weights)):
    print("Weights shape [Layer "+ f"{i}]\n", weights[i])
for i in range(len(biases)):
    print("Biases shape: [Layer "+ f"{i}]\n", biases[i])
"""

activations = [None] * N_LAYERS

plot = Plot(X_train, X_test, y_train, y_test, EPOCHS)

for epoch in range(EPOCHS):
    # Forward propagation
    if epoch % STEP_SIZE == 0:
        LEARNING_RATE *= DECAY_RATE
    acc_train, mse_train, mae_train = plot.append_preds(layers)
    
    if (epoch % 100 == 0):
        print(f"Epoch: {epoch}", "MSE: ", f"{mse_train:.5f}", "R2: ", f"{acc_train:.5f}")

    train_input = X_train
    for i in range(N_LAYERS):
        activations[i], output = layers[i].forward(train_input)
        train_input = activations[i]
    # print(activations[-1])

    for i in reversed(range(N_LAYERS)):
        layers[i].backward(y_train, LEARNING_RATE)

plot_acc_epochs(plot.acc_train, plot.acc_test, EPOCHS)
plot_loss_epochs(plot.mse_train, plot.mse_test, EPOCHS)

""" guesses = 0
for i in range(len(activations[-1])):
    if (activations[-1][i] != y_train[i]):
        guesses += 1
print(f"Incorrect predictions: {guesses}") """

training_test = X_train
for i in range(N_LAYERS):
    activations[i], output = layers[i].forward(training_test)
    training_test = activations[i]

# MSE, MAE, r2 score
mse = f_mse(y_train, activations[-1])
mae = f_mae(y_train, activations[-1])
r2 = f_r2score(y_train, activations[-1])

print()
print(f"Training MSE: {mse}")
print(f"Training MAE: {mae}")
print(f"Training R2 Score: {r2}")
print()

train_input = X_test
for i in range(N_LAYERS):
    activations[i], output = layers[i].forward(train_input)
    train_input = activations[i]

mse = f_mse(y_test, activations[-1])
mae = f_mae(y_test, activations[-1])
r2 = f_r2score(y_test, activations[-1])

print()
print(f"Testing MSE: {mse}")
print(f"Testing MAE: {mae}")
print(f"Testing R2 Score: {r2}")
print()
