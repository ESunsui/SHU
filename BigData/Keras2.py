import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import pandas as pd
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

'''
keras实现神经网络回归模型
'''
'''
# 读取数据
path = 'data.csv'
train_df = pd.read_csv(path)
# 删掉不用字符串字段
dataset = train_df.drop('jh', axis=1)
# df转array
values = dataset.values
# 原始数据标准化，为了加速收敛
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
y = scaled[:, -1]
X = scaled[:, 0:-1]

y = [0.1499, 0.1599, 0.1839, 0.3099, 0.3999, 0.4499, 0.6699, 0.1599, 0.1899, 0.2099, 0.1999, 0.2199, 0.2299, 0.3199,
     0.3499, 0.3899, 0.4599, 0.6499, 0.7299, 0.8999, 0.4799, 0.7499, 0.8299, 1.3999, 1.7999]
X = [0.6346, 0.6696, 0.9955, 1.3353, 1.4195, 1.4944, 1.7925, 0.6968, 0.7746, 0.9868, 1.1595, 1.2668, 1.1974, 1.3864,
     1.6455, 1.6105, 1.8112, 1.8610, 1.9449, 2.1662, 1.6660, 1.9652, 2.1689, 2.4184, 2.5557]
values = y.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
y = scaled[:, -1]
X = scaled[:, 0:-1]
'''
data = pd.read_excel('C:/Users/Fraxinus/Desktop/BigData/GPU.xlsx', 'Sheet3', index_col=0)
#data.dropna(inplace=True)

values = data.iloc[:, 3:16].astype(float)
values.dropna(inplace=True)
print(values)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
X = scaled[:, 0:11]
y = scaled[:, 12]

# 随机拆分训练集与测试集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# 全连接神经网络
model = Sequential()
#input = X.shape[1]
model.add(Dense(units=8, input_dim=11))
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

rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(test_y)
plt.plot(yhat)
pyplot.show()

# 预测y逆标准化
inv_yhat0 = concatenate((test_X, yhat), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat0)
inv_yhat = inv_yhat1[:, -1]
# 原始y逆标准化
test_y = test_y.reshape((len(test_y), 1))
inv_y0 = concatenate((test_X, test_y), axis=1)
inv_y1 = scaler.inverse_transform(inv_y0)
inv_y = inv_y1[:, -1]
# 计算 RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y)
plt.plot(inv_yhat)



