# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:20:38 2021

@author: Fraxinus
"""

import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']

data = pd.read_excel('C:/Users/Fraxinus/Desktop/BigData/GPU.xlsx', 'Sheet2', index_col=0)
data.dropna(inplace=True)
ndata = data[['CUDA核心','加速频率','内存速度','显存带宽','跑分','价格']]
data10 = ndata.iloc[0:6, :]
data16 = ndata.iloc[6:11, :]
data20 = ndata.iloc[12:19, :]
data30 = ndata.iloc[20:24, :]
print(data30)
plt.figure(figsize=(10,10), dpi=300)
pd.plotting.scatter_matrix(ndata, diagonal='kde')
pd.plotting.scatter_matrix(data10, diagonal='kde', color='r')
pd.plotting.scatter_matrix(data16, diagonal='kde', color='g')
pd.plotting.scatter_matrix(data20, diagonal='kde', color='b')
pd.plotting.scatter_matrix(data30, diagonal='kde', color='k')
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.savefig('out10.jpg', dpi=300)