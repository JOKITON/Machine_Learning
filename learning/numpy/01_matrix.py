import numpy as np

print()

# Create basic matrix and print some samples
matrix = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Shape of matrix:", matrix.shape)

# Print some data in the matrix
print(matrix[0, 0])
print(matrix[1, 1])
print(matrix[2, 2])

#* Matrix operations

one = np.ones((1,))
row_ones = np.ones((3,))
col_ones = np.ones((3, 1))

print()
print("Matrix:\n", matrix)
print("Row of ones:\n", row_ones)
print("Column of ones:\n", col_ones)

print()
print("(matrix * row_ones):")
print(matrix + row_ones)

print()
print("(matrix + col_ones):")
print(matrix + col_ones)

print()
print("(matrix * col_ones):")
print(matrix * col_ones)
