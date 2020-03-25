from pyfinn import finn, eiendomspriser, neighborhood, geocode
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow
import talos
from os import path
from notebooks.datasets import clean_and_encode


pd.set_option('display.max_columns', 500)


def fetch_and_prepare(finn_code, header_df):
    ad_data = finn.scrape_ad(finn_code)
    ad_data = finn.interpolate_data_(ad_data)
    ad_data = finn.data_cleaner(ad_data)

    # eiendomspriser
    processed_address = ad_data['Postadresse'].split(',')[0].split('-')[0].split('/')[0]
    sale = eiendomspriser.scrape(processed_address)
    if len(sale['Properties']) > 0:
        ad_data['lat'] = sale['Properties'][0]['Coordinate']['Lat']
        ad_data['lon'] = sale['Properties'][0]['Coordinate']['Lon']
    else:
        ad_data.update(geocode.get_geocode(ad_data['Postadresse'].split(',')[-1]))

    # Nabolag profil
    ad_data.update(neighborhood.scrape(ad_data['lat'], ad_data['lon']))

    df = pd.DataFrame(list([ad_data]))

    # To camelcase and remove æøå
    df.columns = [col.lower().replace(' ', '_').replace('æ', 'ae').replace('ø', 'oe').replace('å', 'aa') for col in
                  df.columns]

    df = clean_and_encode(df)
    df = pd.concat([header_df, df], ignore_index=True)
    df = df[header_df.columns]
    df = df.drop(columns=['totalpris'])
    df = df.reset_index(drop=True)

    # Filling missing values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col].fillna((header_df[col].mean()), inplace=True)

        if 'boligtype' in col and df.at[1, col] != 1 or 'eieform' in col and df.at[1, col] != 1:
            df.at[1, col] = 0

    return df.tail(1)

def XGB_predictor(df):
    xgb_model = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree', colsample_bylevel = 1,
                                 colsample_bytree = 1, gamma = 0, importance_type = 'gain',
                                 learning_rate = 0.1, max_delta_step = 0, max_depth = 9,
                                 min_child_weight = 1, missing = None, n_estimators = 10000, n_jobs = -1,
                                 nthread = None, objective = 'reg:squarederror', random_state = 101, reg_alpha = 2,
                                 reg_lambda = 0.2, scale_pos_weight = 1, seed = None, silent = False, subsample = 1)
    
    xgb_model.load_model('models/model_finn.hdf5')
    
    test_X = df.to_numpy()
    prediction = xgb_model.predict(test_X)
    return prediction[0]

def neural_predictor(test_x, experiment_name=None):
    #Example call: predict(test_x, 'talos_models/03_25_2020_13_21_00.zip')
    
    #Fetches most recent model deployment:
    if experiment_name == None:
        list_of_files = glob.glob('talos_models/*') 
        experiment_name = max(list_of_files, key=os.path.getctime)

    restore_model = Restore(experiment_name+'.zip')
    results = restore_model.model.predict(test_x)
    
    return np.array([inverse_transform(scaler,result) for result in results])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid number of arguments.\n\nUsage:\n$ python predict.py FINNKODE')
        exit(1)

    #Checking for dataset_header.csv
    old_df = None
    if path.exists("dataset_header.csv"):
        old_df = pd.read_csv("dataset_header.csv")
    else:
        print("Can't find 'dataset_header.csv', aborting!")
        exit()

    finn_code = sys.argv[1]
    df = fetch_and_prepare(finn_code, old_df)

    predicted_price_XGB = XGB_predictor(df)
    predicted_price_neural = neural_predictor(df)
    print('The predicted price for the given ad is :\n XGB_prediction: ', int(predicted_price_XGB), 'kr.\n neural_prediction: ', int(predicted_price_neural), 'kr.')

    #print(df)
    #print(df.shape)
