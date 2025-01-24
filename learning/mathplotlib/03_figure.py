import matplotlib.pyplot as plt
import numpy as np

#* Styling Artists

data1, data2 = np.random.randn(2, 100)

fig, ax = plt.subplots(figsize=(10, 6.4))
x = np.arange(len(data1))
ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')
l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)
l.set_linestyle(':')

plt.show()
