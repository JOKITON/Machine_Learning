""" Program to predict car prices using a linear regression model """

import pandas as pd
import config
from colorama import Fore, Back, Style
from plot_utils import crt_diverse_df, crt_plot, crt_diverse_plot
from df_utils import get_thetas_values, display_precision

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

YES_NO = Fore.GREEN + Style.BRIGHT + " (yes" + RESET_ALL + " / " + Fore.RED + Style.BRIGHT + "no): " + RESET_ALL

# Get theta values from thetas.json
THETA0, THETA1 = get_thetas_values()

df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

def make_one_prediction(arg_df, arg_mileage):
    """ Make a prediction for a single mileage value """
    training_mean = arg_df['km'].mean()
    training_std = arg_df['km'].std()

    normalized_x = (arg_mileage - training_mean) / training_std

    price = THETA0 + (THETA1 * normalized_x)

    print(f"\t📊 Predicted price for a car with {arg_mileage} km: " + Fore.GREEN + Style.BRIGHT + f"{price:.2f}\n")

    return price

def make_all_predictions(arg_df):
    """ Make predictions for all mileage values in the dataset """

    training_mean = arg_df['km'].mean()
    training_std = arg_df['km'].std()

    # Initialize a list for predicted prices
    ret_predicted_prices = []

    # Loop over each mileage value to make predictions
    for df_mileage in arg_df['km']:
        # Normalize mileage using the training mean and std
        normalized_x = (df_mileage - training_mean) / training_std

        # Predict price using the normalized mileage
        price = THETA0 + (THETA1 * normalized_x)
        ret_predicted_prices.append(price)

    # Display MSE, MAE and R^2
    display_precision(arg_df, ret_predicted_prices)

    return ret_predicted_prices

def plot_all_predictions(arg_df, arg_predicted_prices):
    """ Plot actual vs predicted prices for all mileage values """

    check = 0
    while check == 0:
        str1 = input("└─> Do you want to create a plot of the data and save it locally?" + YES_NO)
        if str1 == "yes":
            crt_plot(arg_df['km'], arg_df['price'], arg_predicted_prices, 'all')
            while check == 0:
                str2 = input(
                    "\n└─> Do you want to create some plots with different km ranges?" + YES_NO)
                if str2 == "yes":
                    crt_diverse_df(arg_df, arg_predicted_prices)
                    crt_diverse_plot()
                    check = 1
                    break
                if str2 == "no":
                    check = 1
                    break
                else:
                    print("Invalid input. Please try again.\n")
                continue
            break
        if str1 == "no":
            break
        else:
            print("Invalid input. Please try again.\n")
        continue

while 1:
    main_str1 = input("└─> Do you want to predict" + Style.BRIGHT + Fore.LIGHTMAGENTA_EX + " all the prices" + RESET_ALL + " based on the dataset" + YES_NO)
    if main_str1 == "yes":
        predicted_prices = make_all_predictions(df)
        plot_all_predictions(df, predicted_prices)
        break
    if main_str1 == "no":
        break
    print("Invalid input. Please try again.\n")
    continue

while 1:
    main_str2 = input(
        RESET_ALL + "\n└─> Please, input a mileage (only numbers): ")
    if main_str2.isdigit():
        if int(main_str2) < 0 or int(main_str2) > 1000000:
            print(
                Fore.RED + Style.DIM + "Invalid input. Please try again.")
            continue
        mileage = int(main_str2)
        make_one_prediction(df, mileage)
        main_str3 = input(
            Fore.RESET + Back.RESET + Style.RESET_ALL + "└─> Do you want to predict the price for another mileage?" + YES_NO)
        if main_str3 == "yes":
            continue
        if main_str3 == "no":
            print(
                Fore.GREEN + Style.BRIGHT + "👋  Thank you for using my program. Goodbye fellow 42 Coder!\n")
            break
        print(Fore.RED + Style.DIM + "Invalid input. Please try again.")
        continue
    print(Fore.RED + Style.DIM + "Invalid input. Please try again.")
    continue
