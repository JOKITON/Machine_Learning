""" Program that performs linear regression on the car mileage dataset """

import pandas as pd
import config

# Load the dataset
df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

# Normalize mileage values (do this once, not inside the gradient calculation)
mean_mileage = df['km'].mean()
sigma_mileage = df['km'].std()
print("Sigma (Standard deviation of mileage):", sigma_mileage)
df['km'] = (df['km'] - mean_mileage) / sigma_mileage

NUM_ITERATIONS = 5000000  # Number of iterations to repeat
CONVERGENCE_THRESHOLD = 1e-6  # Optional: Stop when updates are very small
LEARNING_RATE = 1e-4  # Increased learning rate

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

def compute_mse(dataframe, theta0, theta1):
    """ Compute the mean squared error """
    predictions = [funct_predict(theta0, theta1, x) for x in dataframe['km']]
    ret_mse = sum((dataframe['price'] - predictions) ** 2) / len(dataframe)
    return ret_mse

# Gradient descent loop
for it in range(NUM_ITERATIONS):
    # Compute the gradients
    tmp0, tmp1 = comp_gradients(THETA0, THETA1, df)

    # Update theta values
    THETA0 -= tmp0
    THETA1 -= tmp1

    # Optional: Check for convergence
    if abs(tmp0) < CONVERGENCE_THRESHOLD and abs(tmp1) < CONVERGENCE_THRESHOLD:
        print(f"Converged after {it+1} iterations!")
        break

    if it % 100 == 0:
        mse = compute_mse(df, THETA0, THETA1)
        print(f"Iteration {it}, MSE = {mse}")

# Output the final results
print("Updated theta0:", THETA0)
print("Updated theta1:", THETA1)
