import argparse, json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from common import build_dataset, validate_dataset, FEATURES

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
args = parser.parse_args()

model = joblib.load(args.model_path)
df = build_dataset(42)
validate_dataset(df)
_, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)
pred = model.predict(test_df[FEATURES])
metrics = {'accuracy': accuracy_score(test_df['target'], pred), 'f1': f1_score(test_df['target'], pred)}
print(json.dumps(metrics, indent=2))
if metrics['accuracy'] < 0.90 or metrics['f1'] < 0.90:
    raise SystemExit('model quality gate failed')
