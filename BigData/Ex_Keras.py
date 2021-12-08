from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 训练集
y = [0.1499, 0.1599, 0.1839, 0.3099, 0.3999, 0.4499, 0.6699, 0.1599, 0.1899, 0.2099, 0.1999, 0.2199, 0.2299, 0.3199,
     0.3499, 0.3899, 0.4599, 0.6499, 0.7299, 0.8999, 0.4799, 0.7499, 0.8299, 1.3999, 1.7999]
x = [0.6346, 0.6696, 0.9955, 1.3353, 1.4195, 1.4944, 1.7925, 0.6968, 0.7746, 0.9868, 1.1595, 1.2668, 1.1974, 1.3864,
     1.6455, 1.6105, 1.8112, 1.8610, 1.9449, 2.1662, 1.6660, 1.9652, 2.1689, 2.4184, 2.5557]
# 待测集
n = 200
w = [i/n*x[-1] for i in range(n)]

# 建模
model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))
# 编译、优化
model.compile(optimizer=Adam(), loss='mse')

for i in range(10):
    # 训练
    model.fit(x, y, epochs=1000, verbose=0)
    print(i, 'loss', model.evaluate(x, y, verbose=0))
    # 预测
    z = model.predict(w)
    # 可视化
    pyplot.subplot(2, 5, i + 1)
    pyplot.xticks(())
    pyplot.yticks(())
    pyplot.scatter(x, y)  # 样本点
    pyplot.scatter(w, z, s=2)  # 预测线
pyplot.show()
