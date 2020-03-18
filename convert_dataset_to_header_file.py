import sys
import pandas as pd
import numpy as np
from notebooks.datasets import clean_and_encode


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid number of arguments.\n\nUsage:\n$ python convert_dataset_to_header_file.py INFILE.csv')
        exit(1)

    file_path = sys.argv[1]

    df = clean_and_encode(pd.read_csv(file_path))

    new_df = pd.DataFrame(columns=df.columns)
    new_df.loc[df.index.max() + 1] = None

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            new_df[col].fillna((df[col].mean()), inplace=True)



    new_df.to_csv(r'dataset_header.csv', index=False)

    print('Saved to file "dataset_header.csv"')
