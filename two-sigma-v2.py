import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def prep_data(market_data):
    # add asset code representation as int (as in previous kernels)
    market_data['assetCodeT'] = market_data['assetCode'].map(lbl)
    market_col = ['assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                        'returnsOpenPrevMktres10']
    # select relevant columns, fillna with zeros (where dropped in previous kernels that I saw)
    # getting rid of time, assetCode (keep int representation assetCodeT), assetName, universe
    X = market_data[market_col].fillna(0).values
    market_data['time'] = pd.to_datetime(market_data['time'], errors='coerce')
    if "returnsOpenNextMktres10" in list(market_data.columns):  # if training data
        up = (market_data.returnsOpenNextMktres10 >= 0).values
        r = market_data.returnsOpenNextMktres10.values
        universe = market_data.universe
        day = market_data.time.dt.date
        assert X.shape[0] == up.shape[0] == r.shape[0] == universe.shape[0] == day.shape[0]
    else:  # observation data without labels
        up = []
        r = []
        universe = []
        day = []
    return X, up, r, universe, day
def visualize_data(df):
    X = df.iloc[:,1].values
    y = df.iloc[:,15].values
    # plotting
    index = np.arange(len(X))
    plt.bar(index, y)
    plt.xlabel('assetCode', fontsize=5)
    plt.ylabel('Open next Mktres 10', fontsize=5)
    plt.xticks(index, X, fontsize=5, rotation=30)
    plt.title('AssetCode - Open next Mktres 10')
    plt.show()
# Feature Scaling
def scale_feature(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test
# Train by LightGBM algorithm
def trainByLightGBM(x_train, y_train, x_test, y_test):
    import lightgbm as lgb
    t = time.time()
    d_train = lgb.Dataset(x_train, label=y_train)
    params = {}
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['max_depth'] = 10
    params['num_leaves'] = 10
    params['learning_rate'] = 0.003
    params['verbosity'] = -1

    print('Begin training...(LightGBM)')
    clf = lgb.train(params, d_train, 100)
    print('Finished, time = ', time.time() - t, 's')
    y_pred = clf.predict(x_test)
    # convert into binary values
    for i in range(0, len(y_pred)):
        if y_pred[i] >= 0.5:  # setting threshold to .5
            y_pred[i] = 1
        else:  
            y_pred[i] = 0
   
    # Accuracy
    accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy of LightGBM:',accuracy)
def trainByXGBoost(x_train, y_train, x_test, y_test):
    from xgboost import XGBClassifier
    xgb_market = XGBClassifier(n_jobs=4, n_estimators=200, max_depth=8, eta=0.1)
    # xgb_market = XGBClassifier(n_jobs=4, n_estimators=100)
    t = time.time()
    print('Begin training...(XGBoost)')
    xgb_market.fit(x_train, y_train)
    print('Finished, time = ', time.time() - t, 's')
    confidence_test = xgb_market.predict_proba(x_test)[:, 1] * 2 - 1
    print('Accuracy of XGBoost:',accuracy_score(confidence_test > 0, y_test))


market_train = pd.read_csv("marketdata_sample.csv")
lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
visualize_data(market_train)