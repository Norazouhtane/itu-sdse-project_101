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


# Read and filter data based on dates
data = pd.read_csv("raw_data.csv")

max_date = "2024-01-31"
min_date = "2024-01-01"

max_date = pd.to_datetime(max_date).date()
min_date = pd.to_datetime(min_date).date()

data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

