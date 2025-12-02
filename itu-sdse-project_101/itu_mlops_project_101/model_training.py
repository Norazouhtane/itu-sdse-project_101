# Import necessary libraries
import datetime
import os
import mlflow
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
data_gold_path = "../data/processed/train_data_gold.csv"
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


# Model training using logistic regression
mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id


with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "../models/lead_model_lr.pkl"

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")

    joblib.dump(value=model, filename=lr_model_path)
    
    mlflow.pyfunc.log_model('model', python_model=LRWrapper(model))

# Save model results
model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)
model_results[lr_model_path] = model_classification_report


# Save model results of both models as json file
model_results_path = "model_results.json"
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)