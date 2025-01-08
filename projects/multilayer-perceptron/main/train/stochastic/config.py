""" config.py """

import os
import sys
from colorama import Fore, Back, Style

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

# Add paths to the system path
DF_UTILS = os.path.join(PWD, 'data/')
UTILS = os.path.join(PWD, 'utils/')
MODEL = os.path.join(PWD, 'model/')
LAYER = os.path.join(PWD, 'layer/')
PLOT = os.path.join(PWD, 'plots/')

sys.path.append(DF_UTILS)
sys.path.append(UTILS)
sys.path.append(MODEL)
sys.path.append(LAYER)
sys.path.append(PLOT)

# Define basic parameters for the model
LEARNING_RATE = 0.035
DECAY_RATE = 0.995
STEP_SIZE = 100

CONVERGENCE_THRESHOLD = 1e-5

# Define the number of epochs for each training method
EPOCHS_STOCHASTIC_1 = 2500
EPOCHS_STOCHASTIC_2 = 5000
EPOCHS_STOCHASTIC_3 = 10000

# Mini-batch size
BATCH_SIZE = 32

# Regularization strength
LAMBDA_REG = 0.01

# General model parameters
N_FEATURES = 30
N_NEURONS = 32
N_HIDDEN_LAYERS = 2 
N_LAYERS = 2 + N_HIDDEN_LAYERS

# Leak ReLU + Sigmoid
LS_SIGMOID_1 = [[N_FEATURES, 32], [32, 20], [20, 10], [10, 1]]
LS_SIGMOID_2 = [[N_FEATURES, 32], [32, 20], [20, 20], [20, 1]]
LS_SIGMOID_3 = [[N_FEATURES, 32], [32, 32], [32, 32], [32, 1]]
LS_SIGMOID_4 = [[N_FEATURES, 256], [256, 128], [128, 64], [64, 1]]

# Leaky ReLU + Softmax
LS_SOFTMAX_1 = [[N_FEATURES, 32], [32, 20], [20, 10], [10, 2]]
LS_SOFTMAX_2 = [[N_FEATURES, 32], [32, 20], [20, 20], [20, 2]]
LS_SOFTMAX_3 = [[N_FEATURES, 32], [32, 32], [32, 32], [32, 2]]
LS_SOFTMAX_4 = [[N_FEATURES, 256], [256, 128], [128, 64], [64, 2]]

# Define paths to datasets
ML_PLOT_PATH = os.path.join(PWD,
	'plots/multilayer-perceptron/')

ML_CLEAN_DATASET = os.path.join(PWD,
	'data/data.csv')

ML_TRAIN_X = os.path.join(PWD,
	'data/X_train.csv')

ML_TRAIN_Y = os.path.join(PWD,
	'data/y_train.csv')

ML_TEST_X = os.path.join(PWD,
	'data/X_test.csv')

ML_TEST_Y = os.path.join(PWD,
	'data/y_test.csv')

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL
