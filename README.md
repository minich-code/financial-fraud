# Fraud Detection ML Pipeline
### AI Engineering Challenge — Submission

---

## Overview

An end-to-end machine learning pipeline for fraud detection in mobile money
transactions. The pipeline ingests raw transaction data, validates it,
engineers features, trains a LightGBM classifier, evaluates performance,
and generates batch predictions with drift monitoring.

**Final Model Performance (on held-out test set):**

| Metric   | Score  |
|----------|--------|
| AUC-ROC  | 0.9890 |
| PR-AUC   | 0.8872 |
| F1       | 0.8158 |

**Decision Thresholds:**

| Threshold | Label      | Action                  |
|-----------|------------|-------------------------|
| >= 0.55   | fraud      | Immediate flag          |
| >= 0.30   | suspicious | Human review queue      |
| < 0.30    | legitimate | Pass through            |

---

## Project Structure

```
financial-fraud/
├── main.py                          # Single entry point — full pipeline
├── workflow.yaml                    # Argo Workflow definition
├── Dockerfile                       # Container image for all stages
├── requirements.txt                 # Pinned Python dependencies
├── pyproject.toml                   # Package definition
├── dvc.yaml                         # DVC pipeline stages
├── params.yaml                      # DVC-tracked parameters
├── predictions.csv                  # Batch predictions on test data (10,000 rows)
├── kind-config.yaml                 # kind cluster config for local Kubernetes
│
├── config/                          # Pipeline configuration
│   ├── data_ingestion.yaml
│   ├── data_validation.yaml
│   ├── schema.yaml
│   ├── feature_engineering.yaml
│   ├── model_training.yaml
│   ├── model_evaluation.yaml
│   └── batch_prediction.yaml
│
├── data/
│   ├── transactions.csv             # Raw training data (50,000 rows)
│   └── test_transactions.json       # Batch prediction input (10,000 rows)
│
├── src/
│   ├── scripts/                     # Argo/standalone entry points
│   │   ├── ingest_data.py
│   │   ├── validate_data.py
│   │   ├── feature_engineering.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── export_model.py
│   ├── pipelines/                   # Pipeline orchestration
│   ├── components/                  # Business logic
│   ├── config_manager/              # Config loading
│   └── utils/                       # Logger, exception handler, commons
│
├── artifacts/                       # Generated outputs (DVC-tracked)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── feature_engineering/
│   ├── model_training/
│   ├── model_evaluation/
│   ├── model_export/
│   └── batch_prediction/
│
└── docs/
    ├── pipeline_architecture.md     # Architecture and design decisions
    └── runbook.md                   # Step-by-step operational guide
```


---
## Additional Documentation

| Document | Description |
|---|---|
| `docs/pipeline_architecture.md` | Full pipeline architecture, design decisions, and data flow diagram |
| `docs/runbook.md` | Step-by-step operational guide for local, Docker, and Argo runs |

> **Note:** `docs/pipeline_architecture.md` contains detailed instructions for running the pipeline through Argo Workflows on a local Kubernetes cluster via kind. If you are evaluating the MLOps deliverables, start there.
---

## Quickstart

### Prerequisites

- Ubuntu 22.04 / 24.04
- Python 3.10
- conda
- Docker
- kind (Kubernetes in Docker)
- kubectl
- Argo CLI

### 1. Clone and set up environment

```bash
git clone <repository-url>
cd financial-fraud

conda create -n financial-fraud python=3.10
conda activate financial-fraud

pip install -r requirements.txt
pip install -e .
```

### 2. Run the pipeline locally

```bash
python main.py
```

### 3. Run with DVC (skips unchanged stages)

```bash
dvc repro
```

### 4. Run batch prediction

```bash
python src/pipelines/pip_06_batch_prediction.py
# Output: artifacts/batch_prediction/predictions.csv
```

### 5. View MLflow experiment tracking

```bash
mlflow ui --port 5000
# Open http://localhost:5000
# Experiment: fraud-detection
```

---

## Docker

### Build image

```bash
docker build -t fraud-detection-pipeline:latest .
```

### Run full pipeline in container

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  fraud-detection-pipeline:latest
```

### Run a single stage

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  fraud-detection-pipeline:latest \
  python src/scripts/validate_data.py
```

---

## Argo Workflows (Local Kubernetes via kind)

### One-time cluster setup

```bash
# Create cluster
kind create cluster --name fraud-detection --config kind-config.yaml

# Install Argo Workflows
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.10/install.yaml
kubectl wait --for=condition=ready pod --all -n argo --timeout=180s

# Grant permissions
kubectl create rolebinding argo-default-admin \
  --clusterrole=admin \
  --serviceaccount=argo:default \
  --namespace=argo
```

### Load image into cluster

```bash
kind load docker-image fraud-detection-pipeline:latest --name fraud-detection
```

### Start Argo UI (keep this terminal open)

```bash
kubectl -n argo port-forward deployment/argo-server 2746:2746
# Open https://localhost:2746
```

### Submit pipeline

```bash
# Lint first
argo lint workflow.yaml

# Submit and watch
argo submit workflow.yaml --watch -n argo
```

### After a reboot

```bash
# Restart cluster
docker start fraud-detection-control-plane

# Verify pods are running
kubectl get pods -n argo
```

---

## Pipeline Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| Data Ingestion | `pip_01_data_ingestion.py` | `data/transactions.csv` | `transactions.parquet` |
| Data Validation | `pip_02_data_validation.py` | `transactions.parquet` | `validated_transactions.parquet` |
| Feature Engineering | `pip_03_feature_engineering.py` | `validated_transactions.parquet` | `X_train/X_test/y_train/y_test`, `pipeline.joblib` |
| Model Training | `pip_04_model_training.py` | `X_train`, `y_train` | `lgb_model.joblib` |
| Model Evaluation | `pip_05_model_evaluation.py` | `lgb_model.joblib`, `X_test`, `y_test` | `evaluation_report.json`, plots |
| Batch Prediction | `pip_06_batch_prediction.py` | `test_transactions.json` | `predictions.csv` |

---

## Feature Engineering

Features engineered from raw transaction data:

| Feature | Description |
|---|---|
| `log_amount` | Log-transformed transaction amount |
| `is_high_value` | Flag for transactions > 10,000 |
| `amount_vs_sender_avg` | Amount relative to sender historical average |
| `balance_drain_rate` | Proportion of sender balance spent |
| `sender_balance_change` | Balance difference before and after |
| `receiver_balance_change` | Receiver balance difference |
| `balance_discrepancy` | Flag for amount/balance mismatch |
| `dist_from_nairobi` | Haversine distance from Nairobi |
| `is_outside_kenya` | Geographic anomaly flag |
| `hour`, `is_night`, `is_weekend` | Temporal features |
| `sender_total_tx` | Sender lifetime transaction count |
| `sender_unique_recv` | Unique receivers per sender |
| `sender_unique_devices` | Device diversity per sender |
| `is_device_switch` | Flag for non-primary device usage |
| `device_unique_senders` | Senders per device (mule device signal) |
| `transaction_type_enc` | Encoded transaction type |

---

## Drift Monitoring

Every batch prediction run checks for distribution shift against the
reference baseline saved during data validation.

| Column | Method | Threshold |
|---|---|---|
| `amount` | PSI | 0.2 |
| `sender_balance_before` | PSI | 0.2 |
| `receiver_balance_before` | PSI | 0.2 |
| `transaction_type` | Categorical PSI | 0.2 |
| `is_fraud` | Rate shift | 5% |

Drift flags are logged to MLflow and written to
`artifacts/batch_prediction/batch_monitoring_report.json`.

---

## Hyperparameter Tuning

Optuna-based hyperparameter search is available but disabled by default.
Enable it when model performance degrades or significant drift is detected:

```yaml
# config/model_training.yaml
run_hyperparameter_search: true
```

Then rerun the pipeline:

```bash
dvc repro
# or
python src/pipelines/pip_04_model_training.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOG_DIR` | `./logs` | Directory for log files |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow tracking server URI |

---

## Known Limitations

1. **Batch-only inference** — no real-time API endpoint. A FastAPI wrapper
   around the loaded model would enable online scoring.

2. **Static PSI baseline** — reference stats are computed once on first run.
   In production, the baseline should refresh on a rolling window.

3. **Local Kubernetes only** — Argo Workflow runs on a local kind cluster.
   Production deployment would target a managed Kubernetes service
   (GKE, EKS, AKS).

4. **Single-node training** — suitable for the current dataset size (50k rows).
   Larger datasets would require distributed training.

---

## Dependencies

Key libraries and versions:

| Library | Version | Purpose |
|---|---|---|
| lightgbm | 4.5.0 | Model training |
| scikit-learn | 1.5.1 | Preprocessing, metrics |
| imbalanced-learn | 0.12.3 | SMOTE oversampling |
| mlflow | 2.16.2 | Experiment tracking |
| optuna | 3.6.1 | Hyperparameter search |
| pandas | 2.2.2 | Data manipulation |
| dvc | — | Pipeline reproducibility |

Full list: `requirements.txt`