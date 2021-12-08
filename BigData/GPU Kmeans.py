import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from pandas import DataFrame

data = pd.read_excel('C:/Users/Fraxinus/Desktop/BigData/GPU.xlsx', 'Sheet3', index_col=0)
values = data.iloc[:, 14:16].astype(float)
values.dropna(inplace=True)
X = DataFrame(values)

estimator = KMeans(n_clusters=6)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签

#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
x4 = X[label_pred == 4]
x5 = X[label_pred == 5]
plt.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c = "green", marker='o', label='label1')
plt.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c = "blue", marker='o', label='label2')
plt.scatter(x3.iloc[:, 0], x3.iloc[:, 1], c = "orange", marker='o', label='label3')
plt.scatter(x4.iloc[:, 0], x4.iloc[:, 1], c = "yellow", marker='o', label='label4')
plt.scatter(x5.iloc[:, 0], x5.iloc[:, 1], c = "purple", marker='o', label='label5')
plt.xlabel('Benchmark')
plt.ylabel('Price')
plt.legend(loc=2)
plt.show()

