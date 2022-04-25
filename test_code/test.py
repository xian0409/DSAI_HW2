# system
import os

# basic
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

# Prophet
# from fbprophet import Prophet

# sklearn evaluate
from sklearn import metrics


train_data = pd.read_csv('training.csv',sep = ',', names=["Open","High","Low","Close"])
test_data = pd.read_csv('testing.csv',sep = ',', names=["Open","High","Low","Close"])
print(train_data)
print(test_data)

# fig=plt.figure(figsize=(20,8))
# plt.xticks(rotation = 90)
# ax1 = fig.add_subplot(111)
# ax1.plot(df.Close,color='red',label='close')
# ax1.plot(df.Open,color='green',label='open')
# # plt.legend()
# # twin 為共享x軸
# # ax2= ax1.twinx()
# # plt.bar(df.date,df.Trading_Volume.astype('int')//1000)
# # ax3 = ax1.twinx()
# plt.show()

# plt.plot(df.index, df.Close, 'b', label='Close')
# plt.plot(df.index, df.Open, 'r', label='Open')
# plt.xlabel('Day')
# plt.xticks(df.index,rotation='vertical') #因兩人賽季數相同，故任選
# plt.title('LeBron James and Dwyane Wade')
# plt.legend(loc = 'lower left')
# plt.show()

# print(df['Close'])