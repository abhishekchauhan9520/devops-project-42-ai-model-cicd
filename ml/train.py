import json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from common import build_dataset, validate_dataset, ARTIFACT_DIR

SEED = int(os.getenv("MODEL_SEED", "42"))
MODEL_NAME = os.getenv("MODEL_NAME", "devops-demo-classifier")


def train() -> dict:
    df = build_dataset(SEED)
    validate_dataset(df)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=SEED)
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=SEED, n_jobs=1)
    model.fit(train_df.drop(columns=["target"]), train_df["target"])
    preds = model.predict(test_df.drop(columns=["target"]))
    metrics = {"accuracy": accuracy_score(test_df["target"], preds), "f1": f1_score(test_df["target"], preds)}
    if metrics["accuracy"] < 0.90 or metrics["f1"] < 0.90:
        raise SystemExit(f"quality gate failed: {metrics}")

    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params({"seed": SEED, "n_estimators": 120, "max_depth": 6})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA", "local"))
        mlflow.set_tag("dataset_version", f"synthetic-{SEED}-v1")
        mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
        print(json.dumps({"run_id": run.info.run_id, "model_name": MODEL_NAME, **metrics}))
    return metrics

if __name__ == "__main__":
    train()
