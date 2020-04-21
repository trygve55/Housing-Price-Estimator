import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow
import talos
from talos import Restore
import os
from os import path
import glob
import sys
from datasets import clean_and_encode
from datasets import load
from sklearn.externals import joblib
from neural_train import inverse_transform
from neural_train import talos_model
from finn_data import evaluate_prediction

from sklearn.preprocessing import MinMaxScaler
from talos import Evaluate
from talos.utils.recover_best_model import recover_best_model

def inverse_transform_df(data):
    #Retrieve scaler:
    scaler = joblib.load('../talos_training/hele_norge.scaler')
    return np.array([inverse_transform(scaler,result) for result in data])

def neural_predictor(test_x, experiment_name=None):
    #Example call: predict(test_x, '../talos_training/03_25_2020_13_21_00.zip')
    
    #Fetches most recent model deployment:
    if experiment_name == None:
        list_of_files = glob.glob('../talos_training/*.zip')
        print(list_of_files) 
        pathname = max(list_of_files, key=os.path.getctime)
        print('Restoring talos-model '+ pathname + '..')

        t = Restore(pathname)

    else:
        t = Restore(experiment_name)
        print('Restoring talos-model '+ experiment_name + '..')

    results = t.model.predict(test_x)
    
    return inverse_transform_df(results), t.model

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print('Invalid number of arguments.\n\nUsage:\n$ python neural_restore_predict test_x test_y PATH/model_name.zip')
    #     exit(1)
    if len(sys.argv) == 2:
        experiment_name = sys.argv[1]
    else:
        experiment_name = None
    
    dummy_scaler = MinMaxScaler()
    train_x, train_y, validation_x, validation_y, test_x, test_y, dummy_scaler = load(f'../input/hele_norge.csv', dummy_scaler)

    #For 10-feature testing:
    features = ['boligtype_Leilighet', 'boligtype_Enebolig', 'bruksareal', 'boligtype_Tomannsbolig', 'postnummer', 'boligtype_Rekkehus', 
    'neighborhood_environment_demographics_housingage_10-30', 'neighborhood_environment_demographics_housingprices_0-2000000', 'neighborhood_environment_demographics_housingage_30-50',
    'eieform_Andel']
    train_x = train_x[features]
    validation_x = validation_x[features]
    test_x = test_x[features]


    predictions, model = neural_predictor(test_x, experiment_name)


    #Formatting:
    predictions = pd.Series(np.ndarray.flatten(predictions))
    prices = pd.Series(np.ndarray.flatten(inverse_transform_df(test_y)))


    evaluate_prediction(predictions, prices)
