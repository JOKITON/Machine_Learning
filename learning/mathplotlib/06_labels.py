import matplotlib.pyplot as plt
import numpy as np

#* Labelling plots

mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)
fig, ax = plt.subplots(figsize=(15, 7.8), layout='constrained')
# the histogram of the data
n, bins, patches = ax.hist(x, 150, density=True, facecolor='C0', alpha=0.75)

# Customize Text properties
ax.set_xlabel('Length [cm]', fontsize=18, color='C2')
ax.set_ylabel('Probability', fontsize=18, color='C1')
ax.set_title('Aardvark lengths\n (not really)')

# Mathematical expressions in text
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
ax.axis([55, 175, 0, 0.03])
ax.grid(True)

plt.show()
