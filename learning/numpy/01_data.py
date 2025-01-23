import numpy as np
import pandas as pd

#* Importing and exporting a CSV

x = pd.read_csv('data.csv', header=0).values
print(x)

x = pd.read_csv('data.csv', usecols=['km']).values
print(x)

a = np.array([[-2.58289208,  0.43014843, -1.24082018, 1.59572603],
              [ 0.99027828, 1.17150989,  0.94125714, -0.14692469],
              [ 0.76989341,  0.81299683, -0.95068423, 0.11769564],
              [ 0.20484034,  0.34784527,  1.96979195, 0.51992837]])

df = pd.DataFrame(a)
print(df)

# df.to_csv('data.csv', index=False)

#* Plotting arrays with Matplotlib
import matplotlib.pyplot as plt

a = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])

x = np.linspace(0, 5, 20)
y = np.linspace(0, 10, 20)
plt.plot(x, y, 'purple') # line
plt.plot(x, y, 'o')      # dots

plt.show()
