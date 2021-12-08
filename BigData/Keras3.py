import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam



data = pd.read_csv('C:/Users/Fraxinus/Desktop/BigData/PriceCompare.csv')
print(data)

values = data.iloc[:,1:4].astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
X = scaled[:, 0:2]
y = scaled[:, 2]
print(X, y)

# 随机拆分训练集与测试集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# 全连接神经网络
model = Sequential()
#input = X.shape[1]
model.add(Dense(units=8, input_dim=2))
model.add(Activation('relu'))
# 隐藏层128
model.add(Dense(16))
model.add(Activation('relu'))
# Dropout层用于防止过拟合
#model.add(Dropout(0.1))
# 隐藏层256
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.01))
# 隐藏层128
model.add(Dense(32))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
# 隐藏层128
model.add(Dense(16))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
# 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
model.add(Dense(1))
# 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
model.compile(loss='mean_squared_error', optimizer=Adam())

# early stoppping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=90, verbose=2)
# 训练
history = model.fit(train_X, train_y, epochs=1000, batch_size=20, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False, callbacks=[early_stopping])
#history = model.fit(train_X, train_y, epochs=1000, batch_size=20, validation_data=(X, y), verbose=2,
#                    shuffle=False, callbacks=[early_stopping])
# loss曲线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)
fig = plt.figure()
ax = plt.axes(projection='3d')
xx = np.arange(0, 1, 0.02)
yy = np.arange(0, 1, 0.02)
mX, mY = np.meshgrid(xx, yy)

Z = np.zeros((50, 50))
for i in range(0, 50, 1):
    for j in range(0, 50, 1):
        tx = float(i/50.0)
        ty = float(j/50.0)
        Z[i][j] = model.predict(np.array([[tx, ty]]))

    print('batch complete' + i)

Z.reshape(2500, 1)
ax.plot_surface(mX, mY, Z, cmap='rainbow')
plt.show()

rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(test_y)
plt.plot(yhat)
pyplot.show()



