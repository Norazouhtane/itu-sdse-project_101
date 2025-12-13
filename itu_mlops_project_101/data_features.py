# Import necessary libraries
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# Define functions for feature engineering
def impute_missing_values(x, method="mean"):
    """
    Impute missing values in a pandas Series.
    
    For numeric values, the mean or median of the column is used, otherwise the mode is used.
    
    Parameters:
        x (pd.Series): pandas Series to impute.
        method (str): Strategy for filling out NaN for numeric values. Default = "mean", otherwise median.
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def create_dummy_cols(df, col):
    """
    Create one-hot encoded columns and drop the original column.
    
    Parameters:
    df (pd.DataFrame): pandas DataFrame to encode.
    col (str): Column to replace with one-hot encoded ones.
    """
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


# Load cleaned data 
data = pd.read_csv("/project/data/interim/data_clean.csv")


# Define categorical and continuous columns
variables = [
    "lead_indicator", 
    "customer_group", 
    "onboarding", 
    "source"
]

for col in variables:
    data[col] = data[col].astype("object")

continuous_columns = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
categorical_columns = data.loc[:, (data.dtypes=="object")]


# Handle outliers by clipping extreme values (mean ± 2std)
continuous_columns = continuous_columns.apply(
    lambda x: x.clip(
        lower = (x.mean()-2*x.std()),
        upper = (x.mean()+2*x.std())
    )
)


# Impute missing data
continuous_columns = continuous_columns.apply(impute_missing_values) 
categorical_columns = categorical_columns.apply(impute_missing_values)


# Scale continuous columns using min-max scaling
scaler = MinMaxScaler()
scaler.fit(continuous_columns)

continuous_columns = pd.DataFrame(scaler.transform(continuous_columns), columns=continuous_columns.columns)


# Recombine continuous and categorical columns
continuous_columns = continuous_columns.reset_index(drop=True)
categorical_columns = categorical_columns.reset_index(drop=True)

data = pd.concat([categorical_columns, continuous_columns], axis=1)


# Drop irrelevant columns
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)


# One-hot encode categorical features
categorical_column_names = ["customer_group", "onboarding", "source"] 
categorical_columns = data[categorical_column_names]
other_columns = data.drop(categorical_column_names, axis=1)

for col in categorical_columns:
    categorical_columns = create_dummy_cols(categorical_columns, col)

data = pd.concat([other_columns, categorical_columns], axis=1)

for col in data:
    data[col] = data[col].astype("float64")


# Save processed data as CSV file
data.to_csv('/project/data/processed/train_data_gold.csv', index=False)