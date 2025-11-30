# Import necessary libraries
import time
import datetime
import json
import mlflow
import pandas as pd

from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


# Define constants
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date


# Get experiment with best f1 score
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]


# Load model results from model training
with open("model_results.json", "r") as f:
    model_results = json.load(f)
results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

# Save model with best f1 sore
best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    # is this used anywhere??