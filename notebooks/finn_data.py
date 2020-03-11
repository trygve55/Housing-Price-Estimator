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

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import sys
sys.path.append('..')


def xgb_train( train_X, train_y, validation_X, validation_y):
    """ Creates and trains a XGB-model on the given data
    Args:
        train_X:       training set input
        train_y:       training set labels
        validation_X = validation set input
        validation_y = validation set labels
    Returns:
        xgb_model    = XGB-model trained on given data
    """
    model_name_wrt = f'../models/model_finn.hdf5'

    xgb_model = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree', colsample_bylevel = 1,
                                 colsample_bytree = 1, gamma = 0, importance_type = 'gain',
                                 learning_rate = 0.1, max_delta_step = 0, max_depth = 9,
                                 min_child_weight = 1, missing = None, n_estimators = 10000, n_jobs = -1,
                                 nthread = None, objective = 'reg:squarederror', random_state = 101, reg_alpha = 2,
                                 reg_lambda = 0.2, scale_pos_weight = 1, seed = None, silent = False, subsample = 1)

    xgb_model.fit(train_X, train_y, eval_set = [(validation_X, validation_y)], eval_metric = 'mae', 
                  early_stopping_rounds = 32, verbose = True)   
    
    xgb_model.save_model(model_name_wrt)
    
    return xgb_model

def feature_importances(xgb_model, train_X):
    """ prints the importances of features 
    Args:
        xgb_model:     XGB-model
        train_X:       training set
    Returns:
    """
    input_features = train_X.columns.values
    feat_imp = xgb_model.feature_importances_
    np.split(feat_imp, len(input_features))
    
    feat_imp_dict = {}
    for i in range(0, len(input_features)):
        feat_imp_dict[feat_imp[i]] = input_features[i]

    sorted_feats = sorted(feat_imp_dict.items(), key = operator.itemgetter(0))
    for i in range(len(sorted_feats) - 1, 0, -1):
        print(sorted_feats[i])

    return


def evaluate_prediction(predictions, test_y): 
    """ prints the importances of features 
    Args:
        predictions: predictions for tast set
        test_y:      test set labels
    Returns:
    """
    test_evaluation = predictions
    test_evaluation['benchmark'] = test_y.median() 
    test_evaluation['target'] = test_y.reset_index(drop=True)
    test_evaluation['difference'] = test_evaluation['pred'] - test_evaluation['target']
    test_evaluation['bench difference'] = test_evaluation['benchmark'] - test_evaluation['target']
    test_evaluation['abs difference'] = abs(test_evaluation['difference'])
    test_evaluation['abs bench difference'] = abs(test_evaluation['bench difference'])
    test_evaluation['difference %'] = (test_evaluation['pred'] / test_evaluation['target'] - 1) * 100
    test_evaluation['bench difference %'] = abs((test_evaluation['benchmark'] / test_evaluation['target'] - 1) * 100)
    
    mean = int(test_evaluation['abs difference'].mean())
    bench_mean = int(test_evaluation['abs bench difference'].mean())
    mean_perc = round(abs(test_evaluation['difference %']).mean(), 2)
    bench_mean_perc = round(abs(test_evaluation['bench difference %']).mean(), 2)
    
    print(f'| mean abs.  difference | our model: {mean} benchmark: {bench_mean}')
    print(f'| mean abs % difference | our model: {mean_perc} % benchmark: {bench_mean_perc} %')

    
    return


if __name__ == "__main__":
    time = datetime.now(pytz.timezone('Europe/Oslo')).strftime('%m.%d.%Y_%H.%M.%S')
    print(f'Notebook initialized execution at {time}.')

    start_time = datetime.now()

    train_X, train_y, validation_X, validation_y, test_X, test_y = datasets.load(f'../input/oslo20000_0403.csv')

    xgb_model = xgb_train(train_X, train_y, validation_X, validation_y) 

    feature_importances(xgb_model, train_X)

    predictions = pd.DataFrame()
    predictions['pred'] = xgb_model.predict(test_X) 

    evaluate_prediction(predictions, test_y)
 