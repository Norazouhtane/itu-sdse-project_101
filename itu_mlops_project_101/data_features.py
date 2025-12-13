# Import necessary libraries
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def impute_missing_values(x, method="mean"):
    """
    Impute missing values in a pandas Series. 
    
    For numeric values, the mean or median of the column is used, otherwise the mode is used. 
    
    Parameters:
        x (pd.Series): pandas Series to impute.
        method (str): Stategy for filling out NaN for numeric values. Default = "mean", otherwise median. 
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


# Drop irrelevant columns
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)


# Handle dummy variables
cat_cols = ["customer_group", "onboarding", "source"] 
cat_vars = data[cat_cols]
other_vars = data.drop(cat_cols, axis=1)

for col in cat_vars:
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")


data.to_csv('/project/data/processed/train_data_gold.csv', index=False)