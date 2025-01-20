""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# Define paths to datasets
FT_LINEAR_REGRESION_THETAS_PATH = os.path.join(PWD, 'main/thetas.json')
FT_LINEAR_REGRESION_PLOT_PATH = os.path.join(PWD, 'main/plots/')
FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN = os.path.join(PWD, 'data/data.csv')
