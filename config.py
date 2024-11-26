# config.py

import os

# Get the current working directory
PWD = os.path.dirname(os.path.abspath(__file__))

# Define paths to datasets
CIFAR10_PATH = os.path.join(PWD, 'datasets/cifar10/cifar10.npz')
# Add more dataset paths as needed

# Example of another dataset path
# OTHER_DATASET_PATH = os.path.join(PWD, 'other_dataset.npz')