""" Program to predict car prices using a linear regression model """

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config

# Final theta values
THETA0 = 6331.733334842021
THETA1 = -1129.7796926855965

df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

# Training normalization parameters (used during model training)
training_mean = df['km'].mean()  # This should be the mean used during training
training_std = df['km'].std()    # This should be the std used during training

print(f"Mean Deviation (Standard Deviation) of mileage values: {training_std:.2f}")
print(f"Training mean of mileage values: {training_mean:.2f}")

# Initialize a list for predicted prices
predicted_prices = []

# Loop over each mileage value to make predictions
for mileage in df['km']:
    # Normalize mileage using the training mean and std
    normalized_x = (mileage - training_mean) / training_std

    # Predict price using the normalized mileage
    price = THETA0 + (THETA1 * normalized_x)
    predicted_prices.append(price)

# Display results
for mileage, price in zip(df['km'], predicted_prices):
    print(f"Mileage: {mileage} km, Predicted Price: {price:.2f}")

# Calculate the Mean Squared Error (MSE) as the loss function
actual_prices = df['price']
mse = sum((actual - predicted) ** 2 for actual, predicted in zip(
    actual_prices, predicted_prices)) / len(actual_prices)

print(f"Mean Squared Error: {mse:.2f}")

mse = mean_squared_error(df['price'], predicted_prices)
mae = mean_absolute_error(df['price'], predicted_prices)
r2 = r2_score(df['price'], predicted_prices)

print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}")

# Plot actual vs predicted prices
below_100k_mileage = []
below_100k_price = []
below_100k_predicted = []

below_200k_mileage = []
below_200k_price = []
below_200k_predicted = []

below_300k_mileage = []
below_300k_price = []
below_300k_predicted = []

for mileage, price, predicted_price in zip(df['km'], df['price'], predicted_prices):
    if mileage <= 100000:
        below_100k_mileage.append(mileage)
        below_100k_price.append(price)
        below_100k_predicted.append(predicted_price)
    elif mileage <= 200000:
        below_200k_mileage.append(mileage)
        below_200k_price.append(price)
        below_200k_predicted.append(predicted_price)
    else:
        below_300k_mileage.append(mileage)
        below_300k_price.append(price)
        below_300k_predicted.append(predicted_price)

plt.scatter(
    below_100k_mileage, below_100k_price, color='blue', label='Actual Prices (<= 100k km)')
plt.scatter(
    below_100k_mileage, below_100k_predicted, color='red', label='Predicted Prices (<= 100k km)')

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs Predicted Prices')

# Save the plot to a file
plt.savefig(config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_100k.png')
print("Plot saved as 'actual_vs_predicted_prices_100k.png'")
# Clear the current plot to avoid overlapping
plt.clf()

plt.scatter(
    below_200k_mileage, below_200k_price, color='green', label='Actual Prices (<= 200k km)')
plt.scatter(
    below_200k_mileage, below_200k_predicted, color='orange', label='Predicted Prices (<= 200k km)')

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs Predicted Prices')

plt.savefig(config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_200k.png')
print("Plot saved as 'actual_vs_predicted_prices_200k.png'")
plt.clf()

plt.scatter(
    below_300k_mileage, below_300k_price, color='purple', label='Actual Prices (> 200k km)')
plt.scatter(
    below_300k_mileage, below_300k_predicted, color='brown', label='Predicted Prices (> 200k km)')

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs Predicted Prices')

plt.savefig(config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_300k.png')
print("Plot saved as 'actual_vs_predicted_prices_300k.png'")
