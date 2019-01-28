import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from itertools import chain

# Setting working enviroment 
path = os.path.expanduser('C:/Users/pvhoang/Desktop/Kaggle/two-sigma')
os.chdir(path)

# Read dât
df_market = pd.read_csv('marketdata_sample.csv', sep=',')
df_news = pd.read_csv('news_sample.csv', sep=',')

print(df_market.info())
print(df_news.info())

def join_market_news(market_train_df, news_train_df):
	# Fix asset codes (str -> list)
	news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")   
	
	# Expand assetCodes
	assetCodes_expanded = list(chain(*news_train_df['assetCodes']))


	assetCodes_index = news_train_df.index.repeat(news_train_df['assetCodes'].apply(len).astype('int32'))
	

	assert len(assetCodes_index) == len(assetCodes_expanded)
	df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
	
	news_cols = ['time', 'assetCodes']
	news_train_df_expanded = pd.merge(df_assetCodes, news_train_df, left_on='level_0', right_index=True, suffixes=(['','_old']))
	news_train_df_expanded = news_train_df_expanded.drop(['level_0'], axis= 1)

	# Join with train
	market_train_df = pd.merge(market_train_df, news_train_df_expanded, how='left', on=['time', 'assetCode'])
	print(market_train_df.info())

	# Free memory
	del news_train_df_expanded
	
	return market_train_df

dates = df_news['time'].values.astype('datetime64[D]')

df_news['time'] = (pd.to_datetime(df_news['time']) - (np.timedelta64(22,'h'))).dt.ceil('1D')

# Round time of market_train_df to 0h of curret day
df_market['time'] = pd.to_datetime(df_market['time']).dt.floor('1D')

# print(df_news.info())

data = join_market_news(df_market, df_news)
data.to_csv("data-merged.csv", index=False)