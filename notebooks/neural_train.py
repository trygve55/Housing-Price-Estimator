import os
import gc
import pytz
import operator
import numpy as np
import pickle as pkl
import glob
import os
from time import sleep
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
#Save scaler:
from sklearn.externals import joblib

import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow import math
import keras
from keras import layers
from keras.models import Sequential
from keras.activations import relu, elu
from keras.layers import Dense, Dropout
import talos
from talos.model import early_stopper
from talos.utils.best_model import activate_model
from talos import Evaluate
from talos import Deploy
from talos import Restore
from talos.utils.recover_best_model import recover_best_model
import logging
import datasets
from XGBoost_train import evaluate_prediction


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)
with np.testing.suppress_warnings() as sup:
    sup.filter(DeprecationWarning, "")
tf.get_logger().setLevel(logging.ERROR)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import sys
sys.path.append('..')

time = datetime.now(pytz.timezone('Europe/Oslo')).strftime('%m.%d.%Y_%H.%M.%S')
print(f'Notebook initialized execution at {time}.')


def talos_model(x_train, y_train, x_val, y_val, parameters):
    model = Sequential()

    model.add(Dense(parameters['neurons_HL1'], 
    input_shape=(train_x.shape[1],), 
    activation=parameters['activation_1'],use_bias=True))

    model.add(Dropout(parameters['dropout1']))

    model.add(Dense(parameters['neurons_HL2'], 
    activation=parameters['activation_2'], use_bias=True))

    model.add(Dropout(parameters['dropout1']))
    
    if parameters['neurons_HL3']:
        model.add(Dense(parameters['neurons_HL3'], 
        activation=parameters['activation_3'], use_bias=True))


    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=parameters['optimizer'], loss=parameters['loss-functions'], 
    metrics=['mse', 'mae'])

    history = model.fit(x_train, y_train,
            batch_size=parameters['batch_size'],epochs=parameters['epochs'],
            verbose=0,validation_data=[x_val, y_val],
            callbacks=[early_stopper(epochs=parameters['epochs'], 
            mode='moderate',monitor='val_loss', patience=25)])
    
    return history, model
    
def inverse_transform(scaler, value):
    mat = np.zeros((1, scaler.scale_.shape[0]))
    mat[0, 0] = value
    return scaler.inverse_transform(mat)[:,0]

def evaluate(scan_model, test_x, test_y):
    eval_model = Evaluate(scan_model)
    results = eval_model.evaluate(np.array(test_x), np.array(test_y), task='continuous',folds=10, metric='loss')
    return np.array([inverse_transform(scaler,result) for result in results])

if __name__ == "__main__":

    
    base_dir = os.getcwd()
    start_time = datetime.now()
    experiment_name = start_time.strftime("%m_%d_%Y_%H_%M_%S")


    
    scaler = MinMaxScaler()
    dataset = 'hele_norge'
    train_x, train_y, validation_x, validation_y, test_x, test_y, scaler = datasets.load(f'../input/'+dataset+'.csv', scaler)
    
    #Save scaler for future predictions:
    joblib.dump(scaler, f'../talos_training/'+ dataset +'.scaler') 


    round_lim = 10
    
    if len(sys.argv) == 2:
        print("10-feature training initialized")
        features = ['boligtype_Leilighet', 'boligtype_Enebolig', 'bruksareal', 'boligtype_Tomannsbolig', 'postnummer', 'boligtype_Rekkehus', 
        'neighborhood_environment_demographics_housingage_10-30', 'neighborhood_environment_demographics_housingprices_0-2000000', 'neighborhood_environment_demographics_housingage_30-50',
        'eieform_Andel']
        parameters = {'activation_1':['relu', 'elu'],
        'activation_2':['relu', 'elu'],
        'activation_3':['relu', 'elu'],
        'optimizer': ['Adam', "RMSprop"],
        'loss-functions': ['mse'],
        'neurons_HL1': [5, 10, 20, 40],
        'neurons_HL2': [5, 10, 20, 40],
        'neurons_HL3': [5, 10, 20, 40, None],
        'dropout1': [0.1, 0.2, 0.3],
        'dropout2': [0.1, 0.2, 0.3],
        'batch_size': [100, 250, 500],
        'epochs': [400, 900]
        }
        train_x = train_x[features]
        validation_x = validation_x[features]
        test_x = test_x[features]
    else:
        print("All-feature training initialized")
        parameters = {'activation_1':['relu', 'elu'],
        'activation_2':['relu', 'elu'],
        'activation_3':['relu', 'elu'],
        'optimizer': ['Adam', "RMSprop"],
        'loss-functions': ['mse'],
        'neurons_HL1': [50, 100, 200, 400],
        'neurons_HL2': [40, 80, 160, 320],
        'neurons_HL3': [40, 80, 160, 320, None],
        'dropout1': [0.1, 0.2, 0.3],
        'dropout2': [0.1, 0.2, 0.3],
        'batch_size': [100, 250, 500],
        'epochs': [400, 900]
        }
     
    print(f'training {round_lim} model(s)..')   
    scan_model = talos.Scan(x=np.array(train_x),
               y=np.array(train_y),
               x_val=np.array(validation_x),
               y_val=np.array(validation_y),
               model=talos_model,
               params=parameters,
               experiment_name="talos_log",
               round_limit=round_lim)

    #Evaluation:           
    results = evaluate(scan_model, test_x, test_y)
    print("mean absolute error on test-set:")
    print(results)


    
    #Deploy model
    os.chdir(os.path.join(base_dir, 'talos_models'))
    Deploy(scan_model,experiment_name, metric='val_mae')
    os.chdir(base_dir)
    
   
    #Evaluate predictions on test-set:
    best_model = scan_model.best_model('val_mae')
    predictions = best_model.predict(test_x)
    #Formatting:
    predictions = pd.Series(np.ndarray.flatten(predictions))
    prices = pd.Series(np.ndarray.flatten(inverse_transform_df(test_y)))

    evaluate_prediction(predictions, prices)
    
    
     
