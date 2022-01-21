import os
import pandas as pd

def load_data(data_dir:str,
              target:str,
              verbose:bool):

    data = pd.read_csv(os.path.join(data_dir, "winequality-red.csv"))

    ## Null value check & Preprocessing
    colnames = list(data.columns)

    # nonempty
    colnames_empty = [i for i in colnames if data[i].isna().sum() == len(data)]
    if verbose:
        print("empty col names : ", colnames_empty)
    data_nonemp = data.drop(colnames_empty, axis=1).copy()

    # nonnan
    colnames_nan = [i for i in list(data_nonemp.columns) if data_nonemp[i].isna().sum() > 0]
    if verbose:
        print("NaN col names : ", colnames_empty)
    data_nonnan = data_nonemp.drop(colnames_nan, axis=1).copy()

    df = data_nonnan.copy()


    X, y = df[df.columns.difference([target])], df[target]

    return X, y


