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


# Feature selection
data = data.drop(
    [
        "is_active", 
        "marketing_consent", 
        "first_booking", 
        "existing_customer", 
        "last_seen"
    ],
    axis=1
)

data = data.drop(
    [
        "domain", 
        "country", 
        "visited_learn_more_before_booking", 
        "visited_faq"
    ],
    axis=1
)


# Data cleaning
data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

data = data[data.source == "signup"]


# Define categorical and continuous columns
vars = [
    "lead_indicator", 
    "customer_group", 
    "onboarding", 
    "source"
]

for col in vars:
    data[col] = data[col].astype("object")

cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]


# Handle outliers
cont_vars = cont_vars.apply(
    lambda x: x.clip(
        lower = (x.mean()-2*x.std()),
        upper = (x.mean()+2*x.std())
    )
)


# Impute missing data
cont_vars = cont_vars.apply(impute_missing_values) 
cat_vars = cat_vars.apply(impute_missing_values)


# Standardize data
scaler = MinMaxScaler()
scaler.fit(cont_vars)

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)


# Combine data
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)

data = pd.concat([cat_vars, cont_vars], axis=1)


# Save train data to file
data.to_csv('train_data_gold.csv', index=False)