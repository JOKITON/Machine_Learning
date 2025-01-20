""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

NUM_ITERATIONS = 3000  # Number of iterations to repeat
CONVERGENCE_THRESHOLD = 1e-6  # Threshold for convergence
LEARNING_RATE = 1e-1

# Define paths to datasets
FT_LINEAR_REGRESION_THETAS_PATH = os.path.join(PWD, 'data/thetas.json')
FT_LINEAR_REGRESION_PLOT_PATH = os.path.join(PWD, 'plots/')
FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN = os.path.join(PWD, 'data/data.csv')
