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
    

# Define date and path for the experiment
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "train_data_gold.csv"
experiment_name = current_date


# Create mlruns directories and set experiment
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)


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


# Model training using XGBoost + RandomForest
model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": [
        "reg:squarederror", 
        "binary:logistic", 
        "reg:logistic"
    ],
    "eval_metric": ["aucpr", "error"]
}

model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)
model_grid.fit(X_train, y_train)


# Save best model based on predictions
y_pred_train = model_grid.predict(X_train)

xgboost_model = model_grid.best_estimator_
xgboost_model_path = "lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

model_results = {
   xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}