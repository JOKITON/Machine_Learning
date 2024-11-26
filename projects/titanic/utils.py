""" Python module for utility functions """

import matplotlib.pyplot as plt
import pandas as pd

# * Normalizing data
def manual_normalize(array):
    ret_array = []
    for value in array:
        ret_array.append((value - min(array)) / (max(array) - min(array)))
    return ret_array

# Print dataframes
def print_head(dftrain, y_train):
    #* Print the head of the databases
    print(dftrain.head())
    print(y_train.head())

def print_columns_raws(dftrain):
    #* Print the number of rows and columns in the training set.
    print(dftrain.describe())

def print_columns(dftrain, dfeval):
    #* Print the number of columns in the training and evaluation sets.
    print(dftrain.shape[0], dfeval.shape[0])

def print_all(dftrain, y_train):
    print_head(dftrain, y_train)
    print_columns_raws(dftrain)
    print_columns(dftrain, y_train)

def show_plots(dftrain, y_train):
    # * Show a plot with different data

    # Shows the age of the people
    dftrain.age.hist(bins=20)
    plt.show()

    # Shows the number of male/females
    dftrain.sex.value_counts().plot(kind='barh')
    plt.show()

    # Shows number of siblings and spouses
    dftrain.n_siblings_spouses.hist(bins=20)
    plt.show()

    # Shows people in each class
    dftrain['class'].value_counts().plot(kind='barh')
    plt.show()

    # Sex vs Survival
    pd.concat([dftrain, y_train], axis=1).groupby(
        'sex'
        ).survived.mean().plot(kind='barh').set_xlabel('% survive')
    plt.show()
