# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from kaggle.competitions import twosigmanews
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import lightgbm as lgb

from itertools import chain

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
env = twosigmanews.make_env() # load env
df_market = env.get_training_data()[0] # load only market data
df_news = env.get_training_data()[1]   # load only news data
# Any results you write to the current directory are saved as output.

print(df_market.isnull().sum())

# Fill empty market fields
def fillMarketEmpty(df_market):
    fill_value=-9999.99
    df_market['returnsClosePrevMktres1'] = df_market['returnsClosePrevMktres1'].fillna(fill_value)
    df_market['returnsOpenPrevMktres1'] = df_market['returnsOpenPrevMktres1'].fillna(fill_value)
    df_market['returnsClosePrevMktres10'] = df_market['returnsOpenPrevMktres10'].fillna(fill_value)
    df_market['returnsOpenPrevMktres10'] = df_market['returnsOpenPrevMktres10'].fillna(fill_value)
    return df_market

#Fill empty fields
def fillEmpty(df):
    fill_value=-9999.99
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-9999.99)
    df_return = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    del df
    del imp
    return df_return

# Merge 2 dataset
def merge_market_news(market_train_df, news_train_df):
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
    
    del df_assetCodes
    # Join with train
    market_train_df = pd.merge(market_train_df, news_train_df_expanded, how="left", on=['time', 'assetCode'])

    # Free memory
    del news_train_df_expanded

    return market_train_df

def dropField(df):
    df = df.drop(['assetName', 'sourceTimestamp', 'firstCreated', 'sourceId','headline', 'subjects', 'audiences', 'headlineTag', 'provider'], axis = 1)
    return df

# Drop unnecessary columns in news dataset
df_news = dropField(df_news)

df_market = fillMarketEmpty(df_market)

dates = df_news['time'].values.astype('datetime64[D]')

# If time > 22:00:00 -> move to next day
df_news['time'] = (pd.to_datetime(df_news['time']) - (np.timedelta64(22,'h'))).dt.ceil('1D')

# Round time of market_train_df to 0h of curret day
df_market['time'] = pd.to_datetime(df_market['time']).dt.floor('1D')

#free memory
del dates

# Merge data
data = join_market_news(df_market, df_news)
del df_market
del df_news
print(data.info())

# Start 
# Init train, test and validate data from training data set
train, validate = np.split(data.sample(frac=1), [int(.8*len(data))])
del data

y_train = train['returnsOpenNextMktres10']
X_train = train.drop(['time', 'assetCode', 'universe', 'returnsOpenNextMktres10'], axis = 1)
del train

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(X_train, y_train)

# free memory
del y_train
del X_train

y_validate = validate['returnsOpenNextMktres10']
X_validate = validate.drop(['time', 'assetCode', 'universe', 'returnsOpenNextMktres10'], axis = 1)
# prediction
y_predicted=regr.predict(X_validate)
print(mean_absolute_error(y_validate, y_predicted))
del X_validate
del y_validate
del y_predicted

# Prediction function
def rfr_predictions(market, news, predictions_template_df):
    market = fillMarketEmpty(market)
    news =dropField(news)
    
    dates = news['time'].values.astype('datetime64[D]')

    news['time'] = (pd.to_datetime(news['time']) - (np.timedelta64(22,'h'))).dt.ceil('1D')

    # Round time of market_train_df to 0h of curret day
    market['time'] = pd.to_datetime(market['time']).dt.floor('1D')
    
    del dates

    # Join data
    test = join_market_news(market, news)
    del market
    del news
    
    X_test = test.drop(['time', 'assetCode'], axis = 1)
    # Fill empty fields
    X_test = fillEmpty(X_test)
    del test
    
    # Predicting
    y_predicted=regr.predict(X_test)
    # Converting into the confidence value, from -1 to 1
    predictions_template_df.confidenceValue = y_predicted

# Apply to predict data
days = env.get_prediction_days()

# Generate the predictions
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    print(market_obs_df.info())
    print(news_obs_df.info())
    rfr_predictions(market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions_template_df)
print('Prediction finished!')
env.write_submission_file()

print([filename for filename in os.listdir('.') if '.csv' in filename])