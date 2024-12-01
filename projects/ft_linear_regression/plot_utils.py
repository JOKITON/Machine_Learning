""" Functions to create plots for the linear regression project. """

import matplotlib.pyplot as plt
import config

def crt_plot(df_mileage, df_real_price, df_predicted_price, mile_frame):
    """ Create a plot of actual vs predicted prices """
    plt.scatter(
        df_mileage,
        df_real_price,
        color='blue',
        label='Actual Prices (<=' + mile_frame + ' km)')
    plt.scatter(
        df_mileage,
        df_predicted_price,
        color='red',
        label='Predicted Prices (<= ' + mile_frame + ' km)')

    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Actual vs Predicted Prices')

    # Save the plot to a file
    plt.savefig(
        config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_' + mile_frame + '.png')
    print("Plot saved as 'actual_vs_predicted_prices_" + mile_frame + ".png'")
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
