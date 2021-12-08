from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

from mydataset import MyDataset
from model import Model

data = pd.read_csv("used_car_train_20200313.csv", sep=" ")
print(data.info())
data["notRepairedDamage"] = data["notRepairedDamage"].apply(lambda x: x.replace(r'-', '0'))
data = data.astype('float32')
data = data.iloc[0:10000, :]

print(data.info())
print(np.isnan(data).any())
data.dropna(inplace=True)
print(data.info())

data_X = data[["regDate", "model", "brand", "bodyType", "fuelType", "gearbox",
               "power", "kilometer", "notRepairedDamage", "regionCode", "seller", "offerType",
               "creatDate", "v_0", "v_1", "v_2", "v_3", "v_4", "v_5", "v_6", "v_7", "v_8", "v_9",
               "v_10", "v_11", "v_12", "v_13", "v_14"]]
data_y = data[["price"]]
data_y = np.ravel(data_y)

# mpl.rcParams['font.sans-serif'] = ['MicroSoft YaHei']
# plt.figure()
# pd.plotting.scatter_matrix(data.iloc[0:1000, :], diagonal='kde')
# plt.xticks(fontsize=5)
# plt.yticks(fontsize=5)
print(data["price"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(data_X, data_y, test_size=0.25, random_state=4)
train_length = X_train.shape[0]
val_length = X_test.shape[0]

i = []
for item in range(val_length):
    i.append(item)

# 训练数据和测试数据进行标准化处理
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 三种集成回归模型进行训练和预测
# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
rfr_y_predict = rfr.predict(X_test)

# 随机森林回归模型评估
print("随机森林回归的默认评估值为：", rfr.score(X_test, y_test))
print("随机森林回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
print("随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(rfr_y_predict)))
print("随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(rfr_y_predict)))
plt.figure()
plt.scatter(i, rfr_y_predict, c='red')
plt.scatter(i, y_test, c='blue')
plt.show()


# 极端随机森林回归
etr = ExtraTreesRegressor()
# 训练
etr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
etr_y_predict = etr.predict(X_test)

# 极端随机森林回归模型评估
print("极端随机森林回归回归的默认评估值为：", etr.score(X_test, y_test))
print("极端随机森林回归的R_squared值为：", r2_score(y_test, etr_y_predict))
print("极端随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(etr_y_predict)))
print("极端随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(etr_y_predict)))
plt.figure()
plt.scatter(i, etr_y_predict, c='red')
plt.scatter(i, y_test, c='blue')
plt.show()


# 梯度提升回归
gbr = GradientBoostingRegressor()
# 训练
gbr.fit(X_train, y_train.ravel())
# 预测 保存预测结果
gbr_y_predict = gbr.predict(X_test)

# 极端随机森林回归模型评估
print("梯度提升回归的默认评估值为：", gbr.score(X_test, y_test))
print("梯度提升回归的R_squared值为：", r2_score(y_test, gbr_y_predict))
print("梯度提升回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(gbr_y_predict)))
print("梯度提升回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(gbr_y_predict)))
plt.figure()
plt.scatter(i, gbr_y_predict, c='red')
plt.scatter(i, y_test, c='blue')
plt.show()
