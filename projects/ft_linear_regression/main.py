""" Program that performs linear regression on the car mileage dataset """

import pandas as pd
import config
from colorama import Fore, Back, Style
from df_utils import set_thetas_values
from tqdm import tqdm

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

print()

# Load the dataset
df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

# Normalize mileage values (do this once, not inside the gradient calculation)
mean_mileage = df['km'].mean()
sigma_mileage = df['km'].std()
# print("Sigma (Standard deviation of mileage):", sigma_mileage)

df['km'] = (df['km'] - mean_mileage) / sigma_mileage

NUM_ITERATIONS = 3000  # Number of iterations to repeat
CONVERGENCE_THRESHOLD = 1e-13  # Threshold for convergence
LEARNING_RATE = 1e-1

THETA0 = 0
THETA1 = 0

def funct_predict(theta0, theta1, mileage):
    """ Main function to predict the price of a car """
    return theta0 + (theta1 * mileage)

def comp_gradients(theta0, theta1, dataframe):
    """ Compute the gradients of the cost function """
    m = len(dataframe)
    sum0 = 0
    sum1 = 0
    for i in range(m):
        sum0 += funct_predict(
            theta0, theta1, dataframe['km'][i]) - dataframe['price'][i]

        sum1 += (funct_predict(
            theta0, theta1, dataframe['km'][i]) - dataframe['price'][i]) * dataframe['km'][i]
    ret_tmp0, ret_tmp1 = sum0 / m, sum1 / m
    ret_tmp0 = LEARNING_RATE * ret_tmp0
    ret_tmp1 = LEARNING_RATE * ret_tmp1
    return ret_tmp0, ret_tmp1

def mse(dataframe, theta0, theta1):
    """ Compute the mean squared error """
    predictions = [funct_predict(theta0, theta1, x) for x in dataframe['km']]
    ret_mse = sum((dataframe['price'] - predictions) ** 2) / len(dataframe)
    return ret_mse

# Gradient descent loop
# for it in range(NUM_ITERATIONS):
with tqdm(
    total=NUM_ITERATIONS,
    desc= Style.BRIGHT + "⌛ Please wait for the results...",
    leave=False) as pbar:
    for it in range(NUM_ITERATIONS):
        pbar.update(1)
        # Compute the gradients
        tmp0, tmp1 = comp_gradients(THETA0, THETA1, df)

        # Update theta values
        THETA0 -= tmp0
        THETA1 -= tmp1

        # Optional: Check for convergence
        if abs(tmp0) < CONVERGENCE_THRESHOLD and abs(tmp1) < CONVERGENCE_THRESHOLD:
            mse = mse(df, THETA0, THETA1)
            print("\tIteration " + Fore.LIGHTWHITE_EX
                + Style.BRIGHT + f"{it}" + RESET_ALL
                + ", MSE = " + Style.BRIGHT + f"{mse:3f}" + RESET_ALL)
            print( Fore.GREEN + Style.BRIGHT
                + f"\nConverged after {it+1} iterations!" + RESET_ALL + "\n")
            break

        if it % 100 == 0:
            mse = mse(df, THETA0, THETA1)
            # print(f"\tIteration {it}, MSE = {mse:3f}")

# Output the final results
print("🧮 Results:" + RESET_ALL)

print(Style.BRIGHT + "\t✴️ theta0: " + Fore.LIGHTBLUE_EX
      + Style.BRIGHT + f"{THETA0:3f}" + RESET_ALL)

print(Style.BRIGHT + "\t✴️ theta1: " + Fore.LIGHTCYAN_EX
      + Style.BRIGHT + f"{THETA1:3f}" + RESET_ALL)

print(Style.DIM + "\n📥 Theta values have been saved in "
        + config.FT_LINEAR_REGRESION_THETAS_PATH + RESET_ALL)

set_thetas_values(THETA0, THETA1)
