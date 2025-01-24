import matplotlib.pyplot as plt
import numpy as np

#* Figure

# Figure with no Axes
# fig = plt.figure()

# Figure with one Axes
# fig, ax = plt.subplots()

# Figure with 2x2 subplots
# fig, ax = plt.subplots(2, 2)

# Figure with one Axes on the left and two in the right
""" fig, axs = plt.subplot_mosaic([['left', 'right_top'],
                               ['left', 'right_bottom']]) """

#* Axes & Axis
plt.axes((0.1, 0.1, 0.8, 0.8))

plt.axis([0, 10, 0, 10])

#* Artist

#* Types of inputs to plotting functions

b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)

np.random.seed(19680801)  # seed the random number generator.
data = {'a': np.arange(50),
        'c': np.random.randint(0, 5, 50),
        'd': np.random.randn(50)}
# a: Numbers from 1 to 50
# c: Random numbers from 1 to 50, 50 times
# d: Random numbers from Gaussian distribution, 50 times
data['b'] = data['a'] + 10 * np.random.randn(50)
# b: Numbers from array a + 10 * random numbers from Gaussian distribution
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')

plt.show()
