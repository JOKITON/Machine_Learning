""" config.py """

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Get the current working directory
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Define paths to datasets
CIFAR10_PATH = os.path.join(PWD, 'datasets/cifar10/cifar10.npz')
