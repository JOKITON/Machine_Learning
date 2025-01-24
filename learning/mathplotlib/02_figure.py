import matplotlib.pyplot as plt
import numpy as np

data1, data2, data3, data4 = np.random.randn(4, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(data1, data2, marker='x', color='black')
ax2.plot(data3, data4, marker='o', color='red', linestyle='--')

plt.show()
