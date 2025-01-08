"" "Builds a linear model with Estimators on the Titanic dataset. " ""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, IntegerLookup, Normalization, Dense, Input, Concatenate

# Load your data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Remove survival column from dftrain and dfeval.
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Create preprocessing layers for categorical columns
categorical_preprocessing_layers = {}
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    if dftrain[feature_name].dtype == 'object':
        categorical_preprocessing_layers[feature_name] = StringLookup(vocabulary=vocabulary, output_mode='int')
    else:
        categorical_preprocessing_layers[feature_name] = IntegerLookup(vocabulary=vocabulary, output_mode='int')

# Create preprocessing layers for numeric columns
numeric_preprocessing_layers = {}
for feature_name in NUMERIC_COLUMNS:
    normalizer = Normalization()
    normalizer.adapt(dftrain[feature_name].values)
    numeric_preprocessing_layers[feature_name] = normalizer

def preprocess(features):
    features = features.copy()  # Ensure we don't modify the original DataFrame
    for feature_name in CATEGORICAL_COLUMNS:
        transformed_feature = categorical_preprocessing_layers[feature_name](features[feature_name])
        features[feature_name] = tf.expand_dims(transformed_feature, -1)
    for feature_name in NUMERIC_COLUMNS:
        transformed_feature = numeric_preprocessing_layers[feature_name](features[feature_name])
        features[feature_name] = tf.expand_dims(transformed_feature, -1)
    return features

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        # Preprocess the data
        processed_data_df = preprocess(data_df)
        ds = tf.data.Dataset.from_tensor_slices((dict(processed_data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

# Define the feature columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    feature_columns.append(categorical_preprocessing_layers[feature_name])
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(numeric_preprocessing_layers[feature_name])

# Create a Keras model
inputs = {feature_name: Input(name=feature_name, shape=(1,), dtype=tf.float32) for feature_name in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS}
concatenated_inputs = Concatenate()(list(inputs.values()))
x = Dense(128, activation='relu')(concatenated_inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create the input functions
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Train the model
model.fit(train_input_fn(), epochs=10)

# Evaluate the model
result = model.evaluate(eval_input_fn())
print(result)
