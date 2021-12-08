import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("used_car_train_20200313.csv", sep=" ")
data["notRepairedDamage"] = data["notRepairedDamage"].apply(lambda x: x.replace(r'-', '0'))
del data["seller"]
del data["offerType"]

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
                    'v_11', 'v_12', 'v_13', 'v_14', 'price']
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage',
                        'regionCode', ]

sns.pairplot(data[numeric_features])
plt.show()

Y_data = data['price']

price_numeric = data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending=False), '\n')

f, ax = plt.subplots(figsize=(7, 7))
plt.title('Correlation of Numeric Features with Price', y=1, size=16)
sns.heatmap(correlation, square=True, vmax=0.8)
plt.show()

categorical_features = ['model',
                        'brand',
                        'bodyType',
                        'fuelType',
                        'gearbox',
                        'notRepairedDamage']
for c in categorical_features:
    data[c] = data[c].astype('category')
    if data[c].isnull().any():
        data[c] = data[c].cat.add_categories(['MISSING'])
        data[c] = data[c].fillna('MISSING')


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


f = pd.melt(data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
plt.show()
