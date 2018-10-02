import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Setting working enviroment 
path = os.path.expanduser('C:/Users/pvhoang/Desktop/two-sigma')
os.chdir(path)

# Read dât
df_market = pd.read_csv('marketdata_sample.csv', sep=',')
df_news = pd.read_csv('news_sample.csv', sep=',')

# print(df_market.columns)
# print(df_news.columns)

# Data Processing

df_apple = df_market.loc[df_market['assetName'] == 'Apple Inc']
df_apple['price'] = np.mean((df_apple['close'].values, df_apple['open'].values))

df_apple = df_apple.drop(['close', 'open'], axis=1)
df_apple = df_apple.drop(['assetCode', 'universe'], axis=1) #asset code not important, dont know what universe means in this context
df_apple = df_apple.drop(['returnsClosePrevRaw1',
                         'returnsOpenPrevRaw1',
                         'returnsClosePrevMktres1',
                         'returnsOpenPrevMktres1',
                         'returnsClosePrevRaw10',
                         'returnsOpenPrevRaw10',
                         'returnsClosePrevMktres10',
                         'returnsOpenPrevMktres10',
                         'returnsOpenNextMktres10']
                        , axis=1) # don't know difference between Raw and Mktres, also dont know how relevant

df_news = df_news.drop(['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D', 'noveltyCount7D'], axis=1)
df_news = df_news.drop(['volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D'], axis=1)
df_apple_news = df_news.loc[df_news['assetName'] == 'Apple Inc']

df_apple_news['sentiment'] = df_apple_news['sentimentNeutral'].values - df_apple_news['sentimentNegative'].values + df_apple_news['sentimentPositive'].values

df_apple_news = df_apple_news.drop(['sentimentNeutral', 'sentimentNegative', 'sentimentPositive'], axis=1)

#we dont need the sign of the sentiment as we use a scalar. Have to investigave is a scalar is precise enough vs just using the sign. 
df_apple_news = df_apple_news.drop(['sentimentClass'], axis=1) 

df_apple_news = df_apple_news.drop(['sentenceCount'], axis=1) #wordCount should contain at least similiar information as sentenceCount
df_apple_news = df_apple_news.drop(['assetCodes'], axis=1) #we use assetName

print(df_apple)
print("="*80)
print(df_apple_news.columns)