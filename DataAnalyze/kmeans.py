from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

path = "dataforkmeans.txt"

with open(path, 'r') as f:
    data = f.read()
    data = data.replace('[', '')
    data = data.replace(']', '')
    with open ("data.txt", 'w') as s:
        s.write(data)

df =  pd.read_csv('data.txt', header=None)
df = df.iloc[:, 0:2]
print (df)

estimator = KMeans(n_clusters=4)
estimator.fit(df)
label = estimator.labels_

x0 = df[label == 0]
x1 = df[label == 1]
x2 = df[label == 2]
x3 = df[label == 3]

plt.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c='red', label='0')
plt.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c='green', label='0')
plt.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c='blue', label='0')
plt.scatter(x3.iloc[:, 0], x3.iloc[:, 1], c='yellow', label='0')

plt.show()



class K_Means():
    def __init__(self, k=2) -> None:
        self.k = k
        self.max_epoch = 100
        self.ecount = 20

    def fit(self, xdata):
        count = 0
        self.prev = {}
        self.centers = {}
        for i in range(self.k):
            tmp = xdata.iloc[i]
            self.centers[i] = [tmp.iloc[0], tmp.iloc[1]]
            self.prev[i] = [tmp.iloc[0], tmp.iloc[1]]
        for i in range(self.max_epoch):
            clf = {}
            for i in range(self.k):
                clf[i] = []
            for i in range(xdata.shape[0]):
                dist = []
                for j in range(self.k):
                    x = pow(self.centers[j][0]-xdata.iloc[i, 0], 2)
                    y = pow(self.centers[j][1]-xdata.iloc[i, 1], 2)
                    distance = math.sqrt(x+y)
                    dist.append(distance)
                classfication = dist.index(min(dist))
                clf[classfication].append([xdata.iloc[i, 0], xdata.iloc[i, 1]])
            for c in self.centers:
                self.centers[c] = np.average(clf[c], axis=0)
            for j in range(self.k):
                if self.centers[j][0] == self.prev[j][0] and self.centers[j][1] == self.prev[j][1]:
                    count = count + 1
                else:
                    count = 0
            if count > self.ecount:
                break
            for c in self.centers:
                self.prev[c] = self.centers[c]

    def predict(self, xdata):
        label = []
        for i in range(xdata.shape[0]):
            dist = []
            for j in range(self.k):
                x = pow(self.centers[j][0]-xdata.iloc[i, 0], 2)
                y = pow(self.centers[j][1]-xdata.iloc[i, 1], 2)
                distance = math.sqrt(x+y)
                dist.append(distance)
            classfication = dist.index(min(dist))
            label.append(classfication)
        return np.array(label)

df =  pd.read_csv('data.txt', header=None)
df = df.iloc[:, 0:2]

optimizer = K_Means(k = 4)
optimizer.fit(df)
label = optimizer.predict(df)

x0 = df[label == 0]
x1 = df[label == 1]
x2 = df[label == 2]
x3 = df[label == 3]

plt.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c='red', label='0')
plt.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c='green', label='0')
plt.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c='blue', label='0')
plt.scatter(x3.iloc[:, 0], x3.iloc[:, 1], c='yellow', label='0')

plt.show()