import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def split(df, scaler_function=None):
    scaler = None

    if scaler_function:
        scaler = scaler_function.fit(df[df.columns])
        df[df.columns] = scaler.transform(df[df.columns])
    
    train_df, validation_df = train_test_split(df, test_size=0.3, random_state=0)
    test_df, validation_df = train_test_split(validation_df, test_size=0.5, random_state=0)

    train_df.reset_index(drop=True)
    test_df.reset_index(drop=True)
    validation_df.reset_index(drop=True)

    target = 'totalpris'

    train_y = train_df[target]
    validation_y = validation_df[target]
    test_y = test_df[target]

    train_df = train_df.drop(target, axis=1)
    validation_df = validation_df.drop(target, axis=1)
    test_df = test_df.drop(target, axis=1)

    train_y.reset_index(drop=True)
    test_y.reset_index(drop=True)
    validation_y.reset_index(drop=True)
    
    if scaler:
        return train_df, train_y, validation_df, validation_y, test_df, test_y, scaler

    return train_df, train_y, validation_df, validation_y, test_df, test_y


def clean_and_encode(df, save_clean=False):
    df = df.fillna(value=0)

    fucked_cols = ['url', 'kommunale_avg.', 'energimerking', 'tomt', 'utleiedel', 'postadresse', 'omkostninger',
                   'omkostninger_uten_dokumentavgift']
    fucked_cols = [col for col in fucked_cols if col in df.columns]
    df = df.drop(fucked_cols, axis=1)
    try: 
        cat_col = ['boligtype', 'eieform']

        for col in cat_col:
            df_dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, df_dummies], axis=1).drop([col], axis=1)
    
    # remove outliers
        len_unclean = len(df.index)
        df = df[ (df['totalpris'] >= 500_000) & (df['totalpris'] <= 25_000_000) ]
        df = df[ (df['soverom'] >= 0) & (df['soverom'] <= 10) ]
        df = df[ (df['bruksareal'] >= 0) & (df['bruksareal'] <= 1000) ]
        df = df[ (df['rom'] >= 0) & (df['rom'] <= 100) ]
        df = df[ (df['felleskost/mnd.'] >= 0) & (df['felleskost/mnd.'] <= 1_000_000) ]
        df = df[ (df['etasje'] >= -2) & (df['etasje'] <= 50) ]
        df = df[ (df['fellesgjeld'] >= 0) & (df['fellesgjeld'] <= 8_000_000) ]
        df = df[ (df['fellesformue'] >= 0) & (df['fellesformue'] <= 3_000_000) ]
        df = df[ (df['lat'] >= 50) & (df['lat'] <= 72) ]
        df = df[ (df['lon'] >= 0) & (df['lon'] <= 32) ]
        print(f'cleaning removed {round(100*(len_unclean - len(df.index))/(len_unclean), 2)} % of the original values')
    except:
        print('unable to clean file')
    
    # to remove nabolag-values
    #df = df[df.columns.drop(list(df.filter(regex='neighborhood')))]
    
    # to remove all but 10 features
    """col_lst = ['totalpris', 'boligtype_Gårdsbruk/Småbruk', 'postnummer', 'primaerrom','eieform_Eier (Selveier)',
    'garderobe', 'tg_0', 'boligtype_Leilighet', 'byggeaar', 'boligtype_Enebolig', 'eieform_Andel',]
    df = df[col_lst]"""

    if save_clean:
        try: 
            df.to_csv('input/clean.csv', index=False)
        except:
            try:
                df.to_csv('../input/clean.csv', index=False)
            except:
                print('unable to save clean.csv')
    return df


def load_df(file, columns=None):
    if columns:
        df = pd.read_csv(file, usecols=columns)
    else:
        df = pd.read_csv(file, usecols=columns)
    df.dropna() # added
    df = clean_and_encode(df, save_clean=True)

    return df


def load(file, scaler_function=None, columns=None):
    df = load_df(file, columns)
    return split(df, scaler_function=scaler_function)



def make_plots(df):
    plt.figure(figsize=(16,8))
    totalpris = pd.Series(df['totalpris'])
    totalpris.plot.hist(grid=True, bins=50, rwidth=0.9,
                    color='#607c8e')
    plt.title('Prisfordeling på boliger i hele datasettet')
    plt.xlabel('Boligpris')
    plt.ylabel('Antall boliger')
    plt.ticklabel_format(useOffset=False)
    plt.grid(axis='y', alpha=0.75)
    plt.ticklabel_format(useOffset=False)
    plt.savefig('../plots/price_hist.png')

    return


if __name__ == "__main__":
    df = load('input/finn.csv')