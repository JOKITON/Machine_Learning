""" Most used metrics for regression models. """

import numpy as np

def compute_mse(_results, _predictions):
    """ Computes the Mean Squared Error for the given inputs, weights, and bias. """
    # Calculate the MSE
    ret_mse = np.mean((_predictions - _results) ** 2)
    return ret_mse

def compute_mae(_results, _predictions):
    """ Computes the Mean Absolute Error for the given predictions and results. """
    return np.mean(np.abs(_predictions - _results))

def compute_r2score(_results, _predictions):
    """ Computes the Mean Absolute Error for the given predictions and results. """
    math1 = np.sum((_results - _predictions) ** 2)
    math2 = np.sum((_results - np.mean(_results)) ** 2)
    result = math1/math2
    result = 1 - result
    return result