import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import scipy.signal

data = pd.read_csv('C:/Users/Fraxinus/Desktop/BigData/dat.csv', index_col=0)
data = DataFrame(data)
data.dropna(axis=1, inplace=True)
data.plot()
#plt.show()
x = data.iloc[:, 0]
print(x)

h = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
a = scipy.signal.convolve(x, h)
a = DataFrame(a[9:-10])
print(a)
a.plot()

plt.show()

