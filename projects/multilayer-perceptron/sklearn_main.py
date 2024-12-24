""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

import config
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
import numpy as np
from df_utils import conv_binary, check_df_errors, normalize_df
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix 

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

LEARNING_RATE = 1e-1
CONVERGENCE_THRESHOLD = 1e-10
N_NEURONS = 4

# Load the dataset
df = pd.read_csv(config.MULTILAYER_PERCEPTRON_CLEAN_DATASET)

# Drop unnecessary columns
df.drop(columns=['id'], inplace=True)

# Extract & Normalize the diagnosis column
y = df.pop('diagnosis')
y = conv_binary(y)
# print(col_diag)

# Handle errors, unnamed columns, and NaN values
check_df_errors(df, False)

#! Do it myself instead of using library functions
# Normalize the feature columns
#? Standardize features by removing the mean and scaling to unit variance
X = normalize_df(df, "min-max")

# Split into training and test sets
print(Fore.GREEN + Style.DIM
    + "Splitting the dataset into training and test sets... (80/20)" + RESET_ALL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Check shapes for train_test_split
print("Shape of X_train & X_test: ", X_train.shape, X_test.shape)
print("Shape of Y_train & Y_test: ",y_train.shape, y_test.shape)
print()

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1000, activation='relu', solver='adam', random_state=42, verbose=True)
mlp.fit(X_train, y_train.values.ravel()) 

predictions = mlp.predict(X_test)
print(mlp.get_params())

print(predictions)

print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))
