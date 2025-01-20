""" Functions to create plots for the linear regression project. """

import matplotlib.pyplot as plt
import config
import numpy as np
from df_utils import get_thetas_values

TRAINING_MEAN = 101066.25
TRAINING_STD = 52674.24560550151

def crt_plot(df_mileage, df_real_price, df_predicted_price, mile_frame):
    """ Create a plot of actual vs predicted prices and the regression line """

    if mile_frame != 'all':
        file_print_bool = 1
    else:
        file_print_bool = 0
    theta0, theta1 = get_thetas_values()

    # Scatter plot of actual and predicted prices
    plt.scatter(
        df_mileage,
        df_real_price,
        color='blue',
        label='Actual Prices (<= ' + mile_frame + ' km)')
    plt.scatter(
        df_mileage,
        df_predicted_price,
        color='red',
        label='Predicted Prices (<= ' + mile_frame + ' km)')

    # Generate a range of mileage values
    mileage_range = np.linspace(min(df_mileage), max(df_mileage), 1000)

    # Normalize the mileage range for the regression function
    normalized_mileage = (mileage_range - TRAINING_MEAN) / TRAINING_STD

    # Compute the regression line using normalized mileage
    regression_line = theta0 + theta1 * normalized_mileage

    # Plot the regression line
    plt.plot(
        mileage_range,
        regression_line,
        color='green',
        label=f'Prediction Line: $\\theta_0$={theta0:.2f}, $\\theta_1$={theta1:.2f}')

    # Force consistent Y-axis limits
    plt.ylim(min(df_real_price) - 500, max(df_real_price) + 500)

    # Plot settings
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Actual vs Predicted Prices')

    # Save the plot to a file
    plt.savefig(
        config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_' + mile_frame + '.png')
    print(
        "📥 Plot saved as 'actual_vs_predicted_prices_" + mile_frame + ".png'")

    if file_print_bool == 0:
        print(
            "📩 File location : " + config.FT_LINEAR_REGRESION_PLOT_PATH + "")
        file_print_bool = 1

    # Clear the current plot to avoid overlapping
    plt.clf()

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

def crt_diverse_df(df, predicted_prices):
    """ Create diverse plots based on mileage ranges """
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

def crt_diverse_plot():
    """ Create diverse plots on mileage ranges going from 0-100k, 100-200k, 200-300k """
    crt_plot(
           below_100k_mileage, below_100k_price, below_100k_predicted, '100k')

    crt_plot(
        below_200k_mileage, below_200k_price, below_200k_predicted, '200k')

    crt_plot(
        below_300k_mileage, below_300k_price, below_300k_predicted, '300k')
