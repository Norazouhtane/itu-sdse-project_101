# Import necessary libraries
import datetime
import os
import json
import pandas as pd
import joblib

from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression


def create_dummy_cols(df, col):
    """Create one-hot encoded columns and drop the original column."""
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df
    

# Define date and path for the experiment
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "/project/data/processed/train_data_gold.csv"
experiment_name = current_date


# Load data 
data = pd.read_csv(data_gold_path)
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


# Split data into train and test
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state=42, 
    test_size=0.15, 
    stratify=y
)


# Model training using logistic regression
model = LogisticRegression()
lr_model_path = "/project/models/model.pkl"

params = {
        'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        'penalty':  ["none", "l1", "l2", "elasticnet"],
        'C' : [100, 10, 1.0, 0.1, 0.01]
}
model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
model_grid.fit(X_train, y_train)

joblib.dump(value=model_grid, filename=lr_model_path)