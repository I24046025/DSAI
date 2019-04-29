import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error

dataset = pd.read_csv('data/PL.csv', index_col=['Date'], parse_dates=['Date'])
holidays = pd.read_csv('data/holidays.csv')

data = pd.DataFrame(dataset.PL.copy())
data.columns = ["y"]

#===========================#
#    Data Preprocessing     #
#===========================#

# Create Lags
def create_lags(data):
    for i in range(1, 15):
        data["lag_{}".format(i)] = data.y.shift(i)
    return data
data = create_lags(data)

# Create weekend and holidays
data.index = pd.to_datetime(data.index, format='%Y/%m/%d')
data["weekday"] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1

# Add Taiwanese holidays 
data['datetime'] = data.index
i,j = 0,0
for holiday in holidays.date:
    holiday = pd.to_datetime(holiday, format='%Y/%m/%d')
    holiday = holiday.strftime("%Y/%m/%d")
    for datetime in data.datetime:
        datetime = datetime.strftime("%Y/%m/%d")
        if datetime == holiday:
            data.iloc[j, 16] = holidays['isHoliday'][i]
        j = j + 1
    i = i + 1
    j = 0
data = data.drop(['datetime'], axis=1)

# Accumulate on is_weekend
data['NOH'] = data['is_weekend']
i = 0
accumulation = 0
for day in data['NOH']:
#     print(day)
    if day == 0:
        accumulation = 0
    elif day == 1:
        accumulation = accumulation + 1
    data.iloc[i, 17] = accumulation
    i = i + 1
    
# convert to float type
data = data.astype(float)

#===========================#
#         Modeling          #
#===========================#

def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X)*(1-test_size))
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    return X_train, X_test, y_train, y_test

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.35)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM
import lightgbm as lgb

train_data=lgb.Dataset(X_train_scaled,label=y_train)
valid_data=lgb.Dataset(X_test_scaled,label=y_test)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'regression',
          'nthread': 4,
          'num_leaves': 34,
          'learning_rate': 0.005,
          'subsample': 0.8,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 1,
          'metric' : 'rmse'
          }
        
#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#===========================#
#        Submission         #
#===========================#

test_features = data.append({'y':0},ignore_index=True)
test_features = create_lags(test_features)
test_features = test_features[-1:]

is_weekend = [0, 0, 0, 1, 1, 1, 1, 0]
NOH = [0, 0, 0, 1, 2, 3, 4, 0]
weekday = [0, 1, 2, 3, 4, 5, 6, 0]

# setup daily features
test_features['weekday'] = weekday[0]
test_features['is_weekend'] = is_weekend[0]
test_features['NOH'] = NOH[0]

# Predict for the first day
test_features.astype(float)
test_features_scaled = scaler.transform(test_features.drop(['y'], axis=1))
predictions_lgbm_prob = lgbm.predict(test_features_scaled).round(decimals=3)
test_features.iloc[0, 0] = predictions_lgbm_prob
test_features = test_features.append({'y':0},ignore_index=True)

# setup features and predict for the remaining days
for i in range(1, 8):
    for j in range(1, 15):
        test_features.iloc[i, j] = test_features.iloc[i-1, j-1]
    print(test_features.iloc[i, 1])
    test_features.iloc[i, 15] = weekday[i]
    test_features.iloc[i, 16] = is_weekend[i]
    test_features.iloc[i, 17] = NOH[i]
    
    test_features.astype(float)
    test_features_scaled = scaler.transform(test_features.drop(['y'], axis=1))
    predictions_lgbm_prob = lgbm.predict(test_features_scaled).round(decimals=3)
    test_features.iloc[i, 0] = predictions_lgbm_prob[-1]
    
    test_features = test_features.append({'y':0},ignore_index=True)

# construct submission dataframe
date = ['20190402', '20190403', '20190404', '20190405', '20190406', '20190407', '20190408']
pl = test_features['y'][1:-1]

submission_dict = {'date':date, 'peak_load(MW)':pl}
submission = pd.DataFrame(submission_dict)

# save as csv file
submission.to_csv("submission.csv", index=False)