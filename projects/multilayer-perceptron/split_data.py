import config
from config import RESET_ALL, N_FEATURES, ML_TRAIN_X, ML_TRAIN_Y, ML_TEST_X, ML_TEST_Y
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
from preprocessing import conv_binary, check_df_errors, normalize_df

# Load the dataset
df = pd.read_csv(config.ML_CLEAN_DATASET)
# Drop unnecessary columns
df.drop(columns=['id'], inplace=True)

# Extract & Normalize the diagnosis column
y = df.pop('diagnosis')
y = conv_binary(y)

# Handle errors, unnamed columns, and NaN values
check_df_errors(df, False)
# Normalize the feature columns
X = normalize_df(df, "z-score")

# Check for any mismatch in the number of features
if (N_FEATURES != X.shape[1]):
    print("Invalid number of features")
    exit()

# Split into training and test sets
print(Fore.GREEN + Style.DIM
    + "Splitting the dataset into training and test sets... (80/20)" + RESET_ALL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Assuming X_train and X_test are NumPy arrays
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)

X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)

# Save X_train and X_test to CSV files locally
X_train_df.to_csv(ML_TRAIN_X, index=False)
y_train_df.to_csv(ML_TRAIN_Y, index=False)

X_test_df.to_csv(ML_TEST_X, index=False)
y_test_df.to_csv(ML_TEST_Y, index=False)

print("X_train, y_train, X_test and y_test saved to CSV files.")
