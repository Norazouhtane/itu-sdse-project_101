# Import necessary libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


# Load data 
data = pd.read_csv("/project/data/processed/train_data_gold.csv")


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