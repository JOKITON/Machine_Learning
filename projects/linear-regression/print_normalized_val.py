""" Python program to print the normalized values of mileage and price """

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
