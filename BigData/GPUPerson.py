import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
series = pd.DataFrame()
df = pd.read_csv("BTCETHComp.csv")
series['Date'] = df['Date'].tolist()
series['BTC'] = df['BTC Price'].tolist()
series['ETH'] = df['ETH Price'].tolist()
series['GPU'] = df['GPU Price'].tolist()
ax = plt.gca()
series.plot(kind='line', x='Date', y='BTC', ax=ax)
series.plot(kind='line', x='Date', y='ETH', ax=ax)
series.plot(kind='line', x='Date', y='GPU', ax=ax)
print("Pearson Correlation Coefficient B_E", pearsonr(series['BTC'], series['ETH']))
print("Pearson Correlation Coefficient E_G", pearsonr(series['ETH'], series['GPU']))
print("Pearson Correlation Coefficient B_G", pearsonr(series['BTC'], series['GPU']))
plt.show()


