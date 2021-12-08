# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:20:42 2021

@author: Fraxinus
"""
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import Model
from keras import losses
from keras.optimizers import Adam
from sklearn import metrics
from sklearn import model_selection
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist


data = pd.read_excel('C:/Users/Fraxinus/Desktop/BigData/GPU.xlsx', 'Sheet2', index_col=0)
data.dropna(inplace=True)
x = data.iloc[:, 14].astype(float)/10000
y = data.iloc[:, 15].astype(float)/10000
print(x,y)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.35, random_state=4)

# 训练集
#y = [.27, .16, .06, .036, .044, .04, .022, .017, .022, .014, .017, .02, .019, .017, .011, .01, .03, .05, .066, .09]
#x = list(range(len(y)))
# 待测集
#n = 20
#w = [i/n*len(y) for i in range(n)]

# 建模
#model = Sequential()
#model.add(Dense(units=1, input_dim=1, activation='sigmoid'))
#model.add(Dense(units=1, activation='sigmoid'))
# 编译、优化
#model.compile(loss=losses.mean_squared_error, optimizer='sgd')
#model.compile(optimizer=Adam(), loss='mse')


'''
第一步：选择模型
'''
model = Sequential()
'''
第二步：构建网络层
'''
model.add(Dense(1, input_shape=(1,))) # 输入层，28*28=784
model.add(Activation('tanh')) # 激活函数是tanh
model.add(Dropout(0.5)) # 采用50%的dropout

model.add(Dense(20)) # 隐藏层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(1)) # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax')) # 最后一层用softmax作为激活函数

'''
第三步：编译
'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数
model.compile(loss='mse', optimizer=sgd) # 使用交叉熵作为loss函数


for i in range(10):
    # 训练
    model.fit(x, y, batch_size=200, epochs=50, shuffle=True, verbose=0, validation_split=0.3)
    model.evaluate(x, y, batch_size=200, verbose=0)
    print(i, 'loss', model.evaluate(x, y, verbose=0))
    # 预测
    z = model.predict(x)
    # 可视化
    pyplot.subplot(2, 5, i + 1)
    pyplot.xticks(())
    pyplot.yticks(())
    pyplot.scatter(x, y)  # 样本点
    pyplot.scatter(x, z, s=2)  # 预测线
pyplot.show()
