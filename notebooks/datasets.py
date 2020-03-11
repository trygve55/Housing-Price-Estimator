import pandas as pd
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 500)


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

    target = 'Totalpris'

    train_y = train_df[target]
    validation_y = validation_df[target]
    test_y = test_df[target]

    train_df = train_df.drop(target, axis=1)
    validation_df = validation_df.drop(target, axis=1)
    test_df = test_df.drop(target, axis=1)

    train_y.reset_index(drop=True)
    test_y.reset_index(drop=True)
    validation_y.reset_index(drop=True)

    return train_df, train_y, validation_df, validation_y, test_df, test_y, scaler


def clean_and_encode(df):
    df = df.fillna(value=0)

    fucked_cols = ['url', 'Kommunale avg.', 'Energimerking', 'Tomt', 'Utleiedel', 'Postadresse', 'Omkostninger',
                   'Omkostninger_uten_dokumentavgift']
    fucked_cols = [col for col in fucked_cols if col in df.columns]
    df = df.drop(fucked_cols, axis=1)

    cat_col = ['Boligtype', 'Eieform']

    for col in cat_col:
        df_dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, df_dummies], axis=1).drop([col], axis=1)

    return df


def load_df(file):
    df = pd.read_csv(file)
    df.dropna() # added
    df = clean_and_encode(df)

    return df


def load(file, scaler_function=None):
    df = load_df(file)
    return split(df, scaler_function=scaler_function)


if __name__ == "__main__":
    df = load('input/finn.csv')
    #print(df.head(50))
