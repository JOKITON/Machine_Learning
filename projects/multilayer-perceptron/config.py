""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Define basic parameters for the model
DECAY_RATE = 0.99  # Decay rate for learning rate
STEP_SIZE = 10 # Step size for learning rate decay
LEARNING_RATE = 1

CONVERGENCE_THRESHOLD = 1e-5
EPOCHS = 183
N_NEURONS = 32  # Number of neurons per layer
N_LAYERS = 4  # Number of layers
DIVIDE_NEURONS = False  # Divide neurons by 2 for each layer

N_FEATURES = 30
LAYER_SHAPE = [[N_FEATURES, 4000], [4000, 2000], [2000, 1000], [1000, 1]]

# Define paths to datasets
MULTILAYER_PERCEPTRON_PLOT_PATH = os.path.join(PWD,
	'plots/multilayer-perceptron/')

MULTILAYER_PERCEPTRON_CLEAN_DATASET = os.path.join(PWD,
	'datasets/multilayer-perceptron/data.csv')

MULTILAYER_PERCEPTRON_DATASET = os.path.join(PWD,
	'datasets/multilayer-perceptron/raw_data.csv')
