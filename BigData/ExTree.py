from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

'''
随机森林回归
极端随机森林回归
梯度提升回归

通常集成模型能够取得非常好的表现
'''

# 1 准备数据
# 读取GPU信息
data = pd.read_excel('C:/Users/Fraxinus/Desktop/BigData/GPU.xlsx', 'Sheet2', index_col=0)
data = data.iloc[0:29, 0:16]
data.dropna(inplace=True)
# 查看数据描述
print(data)



X = data.iloc[:, 3:13].values.astype(float)
y = data.iloc[:, 14].values.astype(float)
#y = data.iloc[:, 15].values.astype(float)
y = np.ravel(y)

# 2 分割训练数据和测试数据
# 随机采样25%作为测试 75%作为训练
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=4)

# 3 训练数据和测试数据进行标准化处理
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 4 三种集成回归模型进行训练和预测
# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
rfr_y_predict = rfr.predict(X_test)

# 极端随机森林回归
etr = ExtraTreesRegressor()
# 训练
etr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
etr_y_predict = rfr.predict(X_test)

# 梯度提升回归
gbr = GradientBoostingRegressor()
# 训练
gbr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
gbr_y_predict = rfr.predict(X_test)

# 5 模型评估
# 随机森林回归模型评估
print("随机森林回归的默认评估值为：", rfr.score(X_test, y_test))
print("随机森林回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
print("随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(rfr_y_predict)))
print("随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(rfr_y_predict)))

# 极端随机森林回归模型评估
print("极端随机森林回归的默认评估值为：", etr.score(X_test, y_test))
print("极端随机森林回归的R_squared值为：", r2_score(y_test, gbr_y_predict))
print("极端随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(gbr_y_predict)))
print("极端随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(gbr_y_predict)))

# 梯度提升回归模型评估
print("梯度提升回归回归的默认评估值为：", gbr.score(X_test, y_test))
print("梯度提升回归回归的R_squared值为：", r2_score(y_test, etr_y_predict))
print("梯度提升回归回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(etr_y_predict)))
print("梯度提升回归回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(etr_y_predict)))
