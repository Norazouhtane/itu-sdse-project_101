# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def impute_missing_values(x, method="mean"):
    """
    Fills out NaN cells with the mean/median of the column.
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x
