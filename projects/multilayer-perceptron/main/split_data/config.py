""" config.py """

import os
import sys
from colorama import Fore, Back, Style

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

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

# General model parameters
N_FEATURES = 30
N_NEURONS = 32
N_HIDDEN_LAYERS = 2 
N_LAYERS = 2 + N_HIDDEN_LAYERS

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
