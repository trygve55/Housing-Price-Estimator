import os
import gc
import pytz
import operator
import numpy as np
import pickle as pkl
import xgboost as xgb
from time import sleep
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import datasets
import finn_data

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import sys
sys.path.append('..')

if __name__ == "__main__":
    xgb_model = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree', colsample_bylevel = 1,
                                 colsample_bytree = 1, gamma = 0, importance_type = 'gain',
                                 learning_rate = 0.1, max_delta_step = 0, max_depth = 9,
                                 min_child_weight = 1, missing = None, n_estimators = 10000, n_jobs = -1,
                                 nthread = None, objective = 'reg:squarederror', random_state = 101, reg_alpha = 2,
                                 reg_lambda = 0.2, scale_pos_weight = 1, seed = None, silent = False, subsample = 1)
    
    xgb_model.load_model('../models/model_finn.hdf5')

    train_X, train_y, validation_X, validation_y, test_X, test_y, scaler = datasets.load(f'../input/finn.csv') 

    predictions = pd.DataFrame()
    predictions['pred'] = xgb_model.predict(test_X)
    print(predictions['pred'][:][1])
    finn_data.evaluate_prediction(predictions,test_y)