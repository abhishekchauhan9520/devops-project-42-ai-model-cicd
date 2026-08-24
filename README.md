# Project 42 — AI/ML Model CI/CD

Production-style MLOps release pipeline for a deterministic scikit-learn model.

## Pipeline

```text
Git
  ↓
Unit tests + data validation
  ↓
Deterministic training
  ↓
Evaluation gate
  ↓
MLflow tracking / registry
  ↓
Candidate alias
  ↓
Inference smoke test
  ↓
Promotion approval
  ↓
Champion alias
  ↓
Containerized serving
```

## What this demonstrates

- Reproducible model training
- Deterministic dataset generation
- Data and schema validation
- Model quality gates
- MLflow experiment tracking
- MLflow Model Registry
- Versioned model artifacts
- Model tags and aliases (`candidate`, `champion`)
- Inference contract tests
- Containerized model serving
- CI security/quality checks
- Guarded production promotion
- Rollback by moving `champion` to a previous model version

Current MLflow guidance uses model aliases and tags for flexible promotion workflows; fixed model stages are deprecated. citeturn422589search1turn422589search0

## Repository layout

```text
app/                  inference service
ml/                   training, validation and promotion code
tests/                unit and contract tests
models/               local development artifacts only
docker/               serving container
.github/workflows/    CI/CD pipeline
scripts/              operator commands
```

## Local training

```bash
python -m pip install -r requirements.txt
python ml/train.py
python ml/evaluate.py --model-path artifacts/model.joblib
```

The CI pipeline uses the same training and evaluation gates before registry promotion.

## Production notes

- Use a database-backed MLflow backend and durable artifact store.
- Restrict registry write/promotion permissions.
- Keep training datasets immutable and versioned.
- Record Git commit, dataset version and training parameters as model metadata.
- Promote by alias instead of changing application code.
- Roll back by restoring the previous champion alias.

## Validation

```bash
python -m unittest discover -s tests -v
```
