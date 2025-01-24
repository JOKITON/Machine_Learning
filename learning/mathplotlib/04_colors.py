import matplotlib.pyplot as plt
import numpy as np

#* Colors

data1, data2 = np.random.randn(2, 1000)

fig, ax = plt.subplots(figsize=(10, 6.4))

ax.scatter(data1, data2, s=50, facecolor='black', edgecolor='m')

plt.show()
