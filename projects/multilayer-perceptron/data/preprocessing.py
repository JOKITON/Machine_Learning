""" Multiple utility functions for dataframes. """

import pandas as pd
from colorama import Fore, Back, Style

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

def conv_binary(col_diagnosis):
    """ Convert the diagnosis column to binary. """
    return col_diagnosis.map({'M': 0, 'B': 1})

def summarize_df(df):
    """ Summarize the dataframe. """
    print(df.describe())

def check_df_errors(df, verbose=True):
    """ Check for errors in the dataframe. """

    #* Dropping Unnamed: 32 as it's a placeholder column
    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)

    if verbose:
        #* Verify the dataframe shape and remaining columns
        print(Style.DIM + f"Shape after dropping unnecessary columns: {df.shape}")
        print(f"Columns after preprocessing: {df.columns.tolist()}" + RESET_ALL)
        print()

    #* Check for NaN or infinite values in the dataset
    if verbose:
        print(Style.DIM + "Are there NaN values in the dataset?")
        print(pd.isnull(df).any().any())

    if pd.isnull(df).any().any():
        handle_nan_values(df)

    if verbose:
        #* Verify any infinite values in the dataset
        print("Are there infinite values in the dataset?")
        print((df == float('inf')).any().any())
        print(RESET_ALL)

def handle_nan_values(df):
    """ Fill remaining NaN values explicitly with 0 """
    df.fillna(0, inplace=True)

    #* Check columns with persistent NaN values
    # print("Columns with NaN values after mean imputation:")
    # print(df[df.columns[df.isnull().any()]].isnull().sum())

    #* Verify no NaN values remain
    # print("Columns with NaN values after filling with 0:")
    # print(df[df.columns[df.isnull().any()]].isnull().sum())

def min_max_normalize(col):
    """ Normalize column using Min-Max normalization. """
    return (col - col.min()) / (col.max() - col.min())

def standarization(col):
    """ Standarize column using Z-score normalization. """
    return( (col - col.mean()) / col.std() )

def normalize_df(df, method="min-max"):
    """ Normalize the dataframe and return it. """
    for col in df.columns:
        if method == "min-max":
            #* Min Max normalization makes data uniform, small ranged & safer for neural networks
            df[col] = min_max_normalize(df[col])
        elif method == "z-score":
            df[col] = standarization(df[col])
    return df
