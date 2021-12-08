import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import model_selection
data = pd.read_csv('PriceCompare.csv')
print(data)

X = data[['Ori', 'Benchmark']]
y = data['Now']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.35, random_state=4)
linregTr = LinearRegression()
linregTr.fit(X_train, y_train)
print(linregTr.intercept_, linregTr.coef_)
y_train_pred = linregTr.predict(X_train)
y_test_pred = linregTr.predict(X_test)
train_err = metrics.mean_squared_error(y_train, y_train_pred)
test_err = metrics.mean_squared_error(y_test, y_test_pred)
print("train err = " + str(train_err), "\ntest err = " + str(test_err))
pred_score = linregTr.score(X_test, y_test)
print("Predict score is " + str(pred_score))
