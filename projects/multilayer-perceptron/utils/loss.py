""" Loss functions for the project. """

import numpy as np

def compute_loss(_results, _predictions):
    """ Computes the loss for the given predictions and results. """
    return np.sum(_results - _predictions)
