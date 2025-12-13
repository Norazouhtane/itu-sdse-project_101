# Import necessary libraries
import pandas as pd
import numpy as np
import subprocess


# Run dvc update and dvc pull to get raw data
subprocess.run( ["dvc", "update", "raw_data.csv"], cwd="/project/data/raw") 
subprocess.run( ["dvc", "pull"], cwd="/project/data/raw")


# Read in data and filter it based on dates
data = pd.read_csv("/project/data/raw/raw_data.csv")

max_date = "2024-01-31"
min_date = "2024-01-01"

max_date = pd.to_datetime(max_date).date()
min_date = pd.to_datetime(min_date).date()

data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]


# Drop unnecessary columns
data = data.drop(
    [
        "is_active", 
        "marketing_consent", 
        "first_booking", 
        "existing_customer", 
        "last_seen",
        "domain", 
        "country", 
        "visited_learn_more_before_booking", 
        "visited_faq"
    ],
    axis=1
)


# Replace empty strings with NaN values and then drop all rows containing NaN
data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])


# Only keep rows where source == "signup"
data = data[data.source == "signup"] 


# Save cleaned data as CSV file
data.to_csv('/project/data/interim/data_clean.csv', index=False)