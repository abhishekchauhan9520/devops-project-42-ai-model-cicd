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
- Inference contract validation
- Containerized model serving
- CI quality gates
- Guarded production promotion
- Rollback by moving `champion` to a previous model version

Current MLflow guidance uses aliases and tags for flexible promotion workflows; fixed model stages are deprecated.

## Repository layout

```text
app/                  inference service
ml/                   training, validation, registry and promotion code
tests/                unit and contract tests
.github/workflows/    CI/CD pipeline
```

## Local training

```bash
python -m pip install -r requirements.txt
python ml/train.py
python ml/evaluate.py --model-path artifacts/model.joblib
```

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
