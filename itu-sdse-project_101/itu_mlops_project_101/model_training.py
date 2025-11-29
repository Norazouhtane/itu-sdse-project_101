# Import necessary libraries
import datetime
import os
import mlflow
import json
import pandas as pd

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


class LRWrapper(mlflow.pyfunc.PythonModel):
    """"""

    def __init__(self, model):
        """"""
        self.model = model

    def predict(self, context, model_input):
        """Predict probabilities for the positive class."""
        return self.model.predict_proba(model_input)[:, 1]