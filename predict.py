from pyfinn import finn, eiendomspriser, neighborhood, geocode
import sys
import pandas as pd
import numpy as np
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

    print(df)
    print(df.shape)
