""" Python program to print the normalized values of mileage and price """

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def print_normalized_val(df):
    """ Normalize mileage values (do this once, not inside the gradient calculation) """
    mean_mileage = df['km'].mean()
    sigma_mileage = df['km'].std()
    print("Sigma (Standard deviation of mileage):", sigma_mileage)
    df['km'] = (df['km'] - mean_mileage) / sigma_mileage

    mean_price = df['price'].mean()
    sigma_price = df['price'].std()
    df['price'] = (df['price'] - mean_price) / sigma_price

    print(df['km'])
    print(df['price'])

def display_results(df, predicted_prices):
    """ Display results of the linear regression model """
    for mileage, price in zip(df['km'], predicted_prices):
        print(f"Mileage: {mileage} km, Predicted Price: {price:.2f}")

def display_precision(df, predicted_prices):
    """ Display the precision of the linear regression model """
    mse = mean_squared_error(df['price'], predicted_prices)
    mae = mean_absolute_error(df['price'], predicted_prices)
    r2 = r2_score(df['price'], predicted_prices)

    print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}")
