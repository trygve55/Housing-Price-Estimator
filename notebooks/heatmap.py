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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import sys
sys.path.append('..')

def price_per_area(test_X):
    predictions = pd.DataFrame()
    predictions['pred'] = xgb_model.predict(test_X)

    latitude = test_X['lat'].to_numpy()
    longitude = test_X['lon'].to_numpy()
    primary_area = test_X['primaerrom'].to_numpy()
    predicted_price = predictions['pred'].to_numpy()

    predicted_area_price = predicted_price/primary_area

    return predicted_area_price, latitude, longitude

def data_cleaner(predicted_area_price, latitude, longitude):
    latitude = np.abs(latitude)
    longitude = np.abs(longitude)
    delete_long = np.where(longitude > 33)[0]
    delete_lat = np.where(latitude < 57)[0]

    for lon in delete_long:
        longitude = np.delete(longitude,lon)
        latitude = np.delete(latitude,lon)
        predicted_area_price = np.delete(predicted_area_price,lon)

    while np.amin(latitude) < 57:
        lat_del = np.where(latitude == np.amin(latitude))[0]
        latitude = np.delete(latitude,lat_del[0])
        longitude = np.delete(longitude,lat_del[0])
        predicted_area_price = np.delete(predicted_area_price,lat_del[0])
    
    while np.amax(predicted_area_price) > 200000:
        price_del = np.where(predicted_area_price == np.amax(predicted_area_price))[0]
        latitude = np.delete(latitude,price_del[0])
        longitude = np.delete(longitude,price_del[0])
        predicted_area_price = np.delete(predicted_area_price,price_del[0])
    
    return predicted_area_price, latitude, longitude


if __name__ == "__main__":
    xgb_model = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree', colsample_bylevel = 1,
                                 colsample_bytree = 1, gamma = 0, importance_type = 'gain',
                                 learning_rate = 0.1, max_delta_step = 0, max_depth = 9,
                                 min_child_weight = 1, missing = None, n_estimators = 10000, n_jobs = -1,
                                 nthread = None, objective = 'reg:squarederror', random_state = 101, reg_alpha = 2,
                                 reg_lambda = 0.2, scale_pos_weight = 1, seed = None, silent = False, subsample = 1)
    
    xgb_model.load_model('../models/model_finn.hdf5')

    train_X, train_y, validation_X, validation_y, test_X, test_y, scaler = datasets.load(f'../input/finn.csv')
    z, x, y = price_per_area(test_X)
    predicted_area_price, latitude, longitude = data_cleaner(z,x,y)
    
    points = plt.scatter(longitude,latitude,marker='.',c=predicted_area_price,cmap="plasma", lw=0.1)
    plt.colorbar(points)
    plt.xlim(0,33) #longitude min max
    plt.ylim(57,72) #latitude min max
    plt.title('Pris/areal heatmap')
    plt.show()
    plt.savefig('heatmap', dpi=2000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
