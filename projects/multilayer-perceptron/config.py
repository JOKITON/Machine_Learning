""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Define paths to datasets

MULTILAYER_PERCEPTRON_PLOT_PATH = os.path.join(PWD,
	'plots/multilayer-perceptron/')

MULTILAYER_PERCEPTRONN_CAR_MILEAGE_TRAIN = os.path.join(PWD,
	'datasets/multilayer-perceptron/data.csv')
