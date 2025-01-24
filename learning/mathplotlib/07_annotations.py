import matplotlib.pyplot as plt
import numpy as np

#* How to annotate a plot
fig, ax = plt.subplots(figsize=(15, 7.8))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(2.5, 1.5),
            arrowprops=dict(facecolor='black', width=2, shrink=0.05))

ax.set_ylim(-2, 2)

plt.show()
