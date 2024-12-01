""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Get the current working directory
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Define paths to datasets
FT_LINEAR_REGRESION_PLOT_PATH = os.path.join(PWD, 'plots/ft_linear_regression/')
FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN = os.path.join(PWD, 'datasets/ft_linear_regression/data.csv')
