""" Check if CUDA is available """

import tensorflow as tf

# print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))
