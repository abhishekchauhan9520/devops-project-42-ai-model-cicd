import argparse, os
import mlflow
from mlflow import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=os.getenv('MODEL_NAME', 'devops-demo-classifier'))
parser.add_argument('--version', required=True)
parser.add_argument('--action', choices=['candidate','champion'], required=True)
args = parser.parse_args()

client = MlflowClient()
alias = args.action
if args.action == 'champion' and os.getenv('CONFIRM_PROMOTION') != 'PROMOTE-CHAMPION':
    raise SystemExit('production promotion requires CONFIRM_PROMOTION=PROMOTE-CHAMPION')
client.set_registered_model_alias(args.model, alias, args.version)
print(f'{args.model}@{alias} -> version {args.version}')
