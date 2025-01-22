import numpy as np

array = np.array([607, 104, 502, 98, 745, 1011])

#* Sorting arrays
print()
print("\n-- Sorting arrays:")
print("Unsorted array;", array)

sorted_ar = np.sort(array)
print("Sorted* array;", sorted_ar)

# Returns the indices that would sort an array
num_sort = np.argsort(array)
print()
print("Indices of sorted array:", num_sort)

#* Concatenating arrays

array2 = np.array([1, 2, 3, 4, 5, 6])
conc_ar = np.concatenate((array, array2))
print("\n\n-- Concatenating arrays:")
print()
print("Concatenated array:", conc_ar)

sorted_ar = np.sort(conc_ar)
print("Sorted* array;", sorted_ar)

array = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[101, 102, 103], [104, 105, 106]])

conc_ar = np.concatenate((array, array2), axis=0)
print()
print("Concatenated array at axis zero:\n", conc_ar)
print("Shape:", conc_ar.shape)

conc_ar = np.concatenate((array, array2), axis=1)
print()
print("Concatenated array at axis one:\n", conc_ar)
print("Shape:", conc_ar.shape)

array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])

print()
print(array_example)
print("Shape:", array_example.shape)
print("Dimensions:", array_example.ndim)
print("Size:", array_example.size)

#* Reshaping arrays
# Arrays need to have the same size to be reshaped

a = np.arange(6)
print("\n\nReshaping arrays:\n")
print("Before reshape:\n", a)

a = np.reshape(a, (6, 1))
print("After reshape:\n", a)

a = np.reshape(a, newshape=(2, 3), order='C')
print("After reshape:\n", a)

#* How to add new axis to an array
print()
print("-- Adding new axis to an array:\n")

a2 = a
print("Shape before:", a2.shape)
print(a2)

a2 = a[np.newaxis, :]
print("Shape after (column):", a2.shape)
print(a2)

a2 = a[:, np.newaxis]
print("Shape after (row):", a2.shape)
print(a2)

a = np.array([1, 2, 3, 4, 5, 6])

#* How to expand dimensions
print()
print("np.expand_dims() function:")
print("Shape before (row):", a.shape)
a2 = np.expand_dims(a, axis=0)
print("Shape after (axis=0):", a2.shape)
print(a2)
a2 = np.expand_dims(a, axis=1)
print()
print("Shape after (axis=1):", a2.shape)
print(a2)

#* Indexing and slicing
print()
print("-- Indexing and slicing")

data = np.array([1, 2, 3])

print("Array before slicing:", data)

data = data[0:2]
print()
print("Array after slicing:", data)
print()

data = data[1:]
print("Array after slicing:", data)
print()

data = data[-2:]
print("Array after slicing:", data)
print()

#* Conditions applied on arrays
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Array without conditions:\n", a)

data = a[a<5]
print("Array with conditions (<5):\n", data)

data = a[a>5]
print("Array with conditions (>5):\n", data)

condition_five = a>5
data = a[condition_five]
print("Array with conditions (>5):\n", data)

divisible_by_2 = a%2==0
data = a[divisible_by_2]
print("Array with conditions (%2):\n", data)

data = a[(a > 2) & (a < 11)]
print("Array with conditions (>2 & <11):\n", data)

five_up = (a > 5) | (a == 5)
data = a[five_up]
print("Array with conditions (>5 | ==5):\n", data)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.nonzero(a > 5)
print()
print("np.nonzero() function:")
print("Array:\n", a)
print("Indices applying condition(>5) :\n", b)

#* Using coordinates for arrays
print()
print("Coordinates:")
coords = list(zip(b[0], b[1]))
for coord in coords:
    print(coord[0], coord[1])
    
print()
print("-- Creating an array with existing data")
print()

a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

#* Stacking arrays
print("Vertical stacking:")
a3 = np.vstack((a1, a2))
print(a3)
print()

print("Horizontal stacking:")
a3 = np.hstack((a1, a2))
print(a3)

#* Splitting arrays
x = np.arange(1, 25).reshape(2, 12)
print()
print("Horizontal splitting:")
print()
print(np.shape(x))
print("Before horiz splitting:\n", x)
print()

data = np.hsplit(x, 3)
print(np.shape(data))
print("After horiz splitting:\n", data)

data = np.vsplit(x, 2)
print(np.shape(data))
print("After vert splitting:\n", data)

#* Copying arrays
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print()
print("Array before modifications:\n", a)
print()

b = a[0, :]
print("Shallow copy of array[0, :]:\n", b)

b[0] = 101
print()
print("Shallow copy of array after mod:\n", b)

print("Original array after mod:\n", a)

b2 = a.copy()
print()
print("Deep copy of array after mod:\n", b2)

b2[0] = 0

print("Deep copy of array after another mod:\n", b2)

print("Original array after mod:\n", a)

#* Basic array operations
print()
print("Basic array operations:\n")

a = np.array([1, 2, 3, 4])
b = np.ones(4)
c = np.array([5, 5, 10, 10])
d = np.array([[1, 10], [2, 20]])

print("Array a:", a)
print("Array b:", b)
print("Array c:", c)
print("Array d:", d)
print()
print("Array (a + b):", a + b)
print("Array (a - b):", a - b)
print("Array (a * c):", a * c)
print("Array (a / c):", a / c)
print()

print("SUM of (a):", np.sum(a))
print("SUM of (c):", np.sum(c))
print("SUM of (b, axis=0):", np.sum(d, axis=0))
print("SUM of (b, axis=1):", np.sum(d, axis=1))
print()

print("MEAN of (a):", np.mean(a))
print("MEAN of (c):", np.mean(c))
print()

#* Broadcasting
data = np.array([1.0, 2.0])
# For broadcasting to happen, the dimensions of the arrays need to be the same or the multiplier needs to be (1)

print("Data in miles:", data)
data *= 1.6
print("Data in kms:", data)

#* More useful array operations

print()
print("Max. value of data:", np.max(data))
print("Min. value of data:", np.min(data))
print("Sum of values of data:", f"{np.sum(data):.1f}")

a = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
              [0.54627315, 0.05093587, 0.40067661, 0.55645993],
              [0.12697628, 0.82485143, 0.26590556, 0.56917101]])

print()
print("Max. value of a:", np.max(a))
print("Min. value of a:", np.min(a))
print("Sum of values of a:", f"{np.sum(a)}")
print()

#* Creating matrices

data = np.array([[1, 2], [3, 4], [5, 6]])
print("Matrix data:\n", data)

print()
print("data[0, 1]:\n", data[0, 1])
print("data[0:3, 1]:\n", data[0:3, 1])
print("data[0:3, 0:1]:\n", data[0:3, 0:1])

print()
print("Max. :", np.max(data))
print("Min. :", np.min(data))
print("Sum :", f"{np.sum(data)}")

print("Max. (axis=0):", np.max(data, axis=0))
print("Max. (axis=1):", np.max(data, axis=1))
