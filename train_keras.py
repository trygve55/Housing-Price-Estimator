import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow import math
import keras
from keras import layers
from keras.models import Sequential
from keras.activations import relu, elu
from keras.layers import Dense, Dropout
from talos.model import early_stopper
from talos.utils.best_model import activate_model
from talos import Evaluate
from talos import Deploy
import talos
import pandas as pd
from notebooks import datasets
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def inverse_transform(scaler, value):
    mat = np.zeros((1, scaler.scale_.shape[0]))
    mat[0, 0] = value
    return scaler.inverse_transform(mat)[:,0]
    


if __name__ == "__main__":
    scaler = StandardScaler()


    train_x, train_y, validation_x, validation_y, test_x, test_y, scaler = datasets.load(f'input/hele_norge.csv', scaler)
    
    if(sys.argv[1]):
        print("10-parameter training initialized")
        features = ['boligtype_Leilighet', 'boligtype_Enebolig', 'bruksareal', 'boligtype_Tomannsbolig', 'postnummer', 'boligtype_Rekkehus', 
        'neighborhood_environment_demographics_housingage_10-30', 'neighborhood_environment_demographics_housingprices_0-2000000', 'neighborhood_environment_demographics_housingage_30-50',
        'eieform_Andel']
        train_x = train_x[features]
        validation_x = validation_x[features]
        test_x = test_x[features]
        
        
    else:
        print("Training with all features..")
        features = None
    
    print('Train', train_x.shape)
    print('validate', validation_x.shape)
    print('Test', test_x.shape)
  
    parameters = {'activation_1':['relu', 'elu', 'sigmoid', 'tanh'],
         'activation_2':['relu', 'elu', 'sigmoid', 'tanh'],
         'activation_3':['relu', 'elu', 'sigmoid', 'tanh'],
         'optimizer': ['Adam', "RMSprop", 'sgd', 'Nadam'],
         'loss-functions': ['mse'],
         'neurons_HL1': [5, 10, 20, 50],
         'neurons_HL2': [5, 10, 20, 50],
         'neurons_HL3': [5, 10, 20, 50, None],
         'dropout1': [0.1, 0.2, 0.3],
         'dropout2': [0.0, 0.1, 0.2, 0.3],
         'batch_size': [100, 250, 500],
         'epochs': [3000]
    }
    '''
    parameters = {'activation_1':'elu',
         'activation_2':'elu',
         'activation_3':'elu',
         'optimizer': "RMSprop",
         'loss-functions': 'mse',
         'neurons_HL1': 5,
         'neurons_HL2': 5,
         'neurons_HL3': None,
         'dropout1': 0.1,
         'dropout2': 0.3,
         'batch_size': 100,
         'epochs': 900
    }
    '''
    def talolos(x_train, y_train, x_val, y_val, parameters):
        model = Sequential()

        model.add(Dense(parameters['neurons_HL1'],
                        input_shape=(train_x.shape[1],),
                        activation=parameters['activation_1'], use_bias=True))

        model.add(Dropout(parameters['dropout1']))
        model.add(Dense(parameters['neurons_HL2'], activation=parameters['activation_2'], use_bias=True))
        model.add(Dropout(parameters['dropout1']))

        if parameters['neurons_HL3']:
            model.add(Dense(parameters['neurons_HL3'], activation=parameters['activation_3'], use_bias=True))

        model.add(Dense(1, activation='relu'))
        model.compile(optimizer=parameters['optimizer'], loss=parameters['loss-functions'], metrics=['mse', 'mae'])

        history = model.fit(x_train, y_train,
                            batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                            verbose=0, validation_data=[x_val, y_val],
                            callbacks=[early_stopper(epochs=parameters['epochs'],
                                                     mode='moderate', monitor='val_loss', patience=40)])

        return history, model
    print(np.shape(train_x))
    print(np.shape(train_y))
    print(np.shape(validation_x))
    print(np.shape(validation_y))
    
    t = talos.Scan(x=np.array(train_x),
                   y=np.array(train_y),
                   x_val=np.array(validation_x),
                   y_val=np.array(validation_y),
                   model=talolos,
                   params=parameters,
                   experiment_name='talos_training',
                   round_limit=10)
                   
    Deploy(t, '10features_1')

    #hist, mod = talolos(train_x, train_y, validation_x, validation_y, parameters)
    
    print(t.data)

    for index in t.data.index:
        t.data.at[index, 'real_error'] = inverse_transform(scaler, t.data.at[index, 'val_mae'])
        t.data.at[index, 'real_error2'] = inverse_transform(scaler, t.data.at[index, 'val_mse'])


    print(t.data.sort_values(by=['real_error']))

    e = Evaluate(t)

    e.evaluate(test_x, test_y, 'continuous', 'val_mse', print_out=True)

    print(e)
