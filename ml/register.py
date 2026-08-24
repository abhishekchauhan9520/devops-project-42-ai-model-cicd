import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import joblib

model_name = os.getenv('MODEL_NAME', 'devops-demo-classifier')
model_path = os.getenv('MODEL_PATH', 'artifacts/model.joblib')
tracking = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(tracking)

model = joblib.load(model_path)
mlflow.set_experiment(model_name)
with mlflow.start_run() as run:
    info = mlflow.sklearn.log_model(model, name='model', registered_model_name=model_name)
    version = MlflowClient().get_latest_versions(model_name)[-1].version
    MlflowClient().set_registered_model_alias(model_name, 'candidate', version)
    print(f'registered {model_name} version={version} candidate={version} run={run.info.run_id}')
