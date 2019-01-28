import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Setting working enviroment 
path = os.path.expanduser('C:/Users/pvhoang/Desktop/two-sigma')
os.chdir(path)

# Read dât
df_market = pd.read_csv('marketdata_sample.csv', sep=',')
df_news = pd.read_csv('news_sample.csv', sep=',')

print(df_market.shape)
# print(df_news.columns)

# Data Processing
dates = pd.to_datetime(df_market['time'].unique())
df_market.groupby('time').count()['universe'].plot(figsize=(12,5), linewidth= 2)

print("There are {} unique investable assets in the whole history.".format(df_market['assetName'].unique().shape[0]))
plt.show()