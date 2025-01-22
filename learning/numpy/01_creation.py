import numpy as np
from numpy import random as rng

#* Array filled with basic number vector
array = np.array([1, 2, 3, 4, 5, 6])
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

# Print some samples of the array
print()
print(array[3], array[1], array[4])

#* Array filled with zeros
array = np.zeros(4)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Array filled with ones
array = np.ones(8)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Empty array
array = np.empty(10)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Array of ranged elements
array = np.arange(10)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Array of specific ranged elements by step
array = np.arange(100, 150, 5)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Array of divided elements in a specified interval
array = np.linspace(0, 100, 21)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Array filled with ones with changed datatype
array = np.ones(8, dtype=np.int8)
# Print basic information about the array
print()
print(array)
print("Number of dimensions:", array.ndim)
print("Number of dimensions:", len(array.shape))
print("Shape of array:", array.shape)
print("Size of array:", array.size)
print("Data type of array:", array.dtype)

#* Creating random numbers

print()
data = rng.randint(5, size=(2, 3))
print("Random array (5, size=(2, 3)):\n", data)

#* Unique values

a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

print()
print("Array (a):\n", a)

data = np.unique(a)
print()
print("Unique values inside array (a):\n", data)
print()

_, data = np.unique(a, return_index=True)
print()
print("Unique indices inside array (a):\n", data)
print()

_, data = np.unique(a, return_counts=True)
print()
print("Unique value counter inside array (a):\n", data)
print()
