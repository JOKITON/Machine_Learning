""" Program to predict car prices using a linear regression model """

import pandas as pd
import config
from plot_utils import crt_diverse_df, crt_plot, crt_diverse_plot
from df_utils import display_precision

# Final theta values
THETA0 = 6331.833333233351
THETA1 = -1129.8079761625243

df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

# Training normalization parameters (used during model training)
training_mean = df['km'].mean()  # This should be the mean used during training
training_std = df['km'].std()    # This should be the std used during training

# Initialize a list for predicted prices
predicted_prices = []

# Loop over each mileage value to make predictions
for mileage in df['km']:
    # Normalize mileage using the training mean and std
    normalized_x = (mileage - training_mean) / training_std

    # Predict price using the normalized mileage
    price = THETA0 + (THETA1 * normalized_x)
    predicted_prices.append(price)

# Display MSE, MAE and R^2
display_precision(df, predicted_prices)

# Plot actual vs predicted prices

CHECK = 0
while CHECK == 0:
    str1 = input("Do you want to create a plot of the data? (yes/no): ")
    if str1 == "yes":
        crt_plot(df['km'], df['price'], predicted_prices, 'all')
        while 1:
            str2 = input("Do you want to create some plots with different km ranges? (yes/no): ")
            if str2 == "yes":
                crt_diverse_df(df, predicted_prices)
                crt_diverse_plot()
                CHECK = 1
                break
            if str2 == "no":
                CHECK = 1
                break
            print("Invalid input. Please try again.")
            continue
    if str1 == "no":
        break
    print("Invalid input. Please try again.")
    continue
