import time
import requests
import json
import csv

time_stamp = int(time.time())
print(f"Now timestamp: {time_stamp}")
# 1367107200
request_link = f"https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical?convert=USD&slug=bitcoin&time_end={time_stamp}&time_start=1367107200"
print("Request link: " + request_link)
r = requests.get(url = request_link)
#print(r.content)
# 返回的数据是 JSON 格式，使用 json 模块解析
content = json.loads(r.content)
#print(type(content))
quoteList = content['data']['quotes']
#print(quoteList)


# 将数据保存为 BTC.csv
# for windows, newline=''
with open('BTC.csv','w' ,encoding='utf8',newline='') as f:
    csv_write = csv.writer(f)
    csv_head = ["Date","Price","Volume"]
    csv_write.writerow(csv_head)

    for quote in quoteList:
        quote_date = quote["time_open"][:10]
        quote_price = "{:.2f}".format(quote["quote"]["USD"]["close"])
        quote_volume = "{:.2f}".format(quote["quote"]["USD"]["volume"])
        csv_write.writerow([quote_date, quote_price, quote_volume])

print("Done")

import pandas as pd

series = pd.DataFrame()
df = pd.read_csv("BTC.csv")
series['Date'] = df['Date'].tolist()
series['Price'] = df['Price'].tolist()
series['Volume'] = df['Volume'].tolist()

import matplotlib.pyplot as plt

ax = plt.gca()
series.plot(kind='line', x='Date', y='Price', ax=ax)
plt.show()

plt.cla()
ax = plt.gca()
series.plot(kind='line', x='Date', y='Volume', ax=ax)
plt.show()

series = pd.DataFrame()
df = pd.read_csv("BTC_Difficulty.csv")
series['Date'] = df['Time'].tolist()
series['ACP'] = df['AverageComputingPower(EH/s)'].tolist()
series['Dif'] = df['Difficulty'].tolist()
reverse_series = series.iloc[::-1]

ax = plt.gca()
reverse_series.plot(kind='line', x='Date', y='ACP', ax=ax)
plt.show()
plt.cla()
ax = plt.gca()
reverse_series.plot(kind='line', x='Date', y='Dif', ax=ax)
plt.show()
