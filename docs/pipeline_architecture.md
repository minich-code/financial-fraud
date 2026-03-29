
# Financial Fraud Detection

A production-ready machine learning pipeline for detecting fraudulent M-Pesa transactions. Built with LightGBM, feature stores for deployment-safe inference, and full MLflow experiment tracking.

---

## Pipeline Architecture

![Fraud Detection Pipeline](docs/images/Fraud_detection.jpg)

The pipeline runs as six sequential stages, each saving its outputs as parquet files
or joblib artifacts that the next stage consumes.

**pip_01 — Data Ingestion** reads the raw `transactions.csv`, applies basic cleaning,
and persists the result as `transactions.parquet` for downstream stages.

**pip_02 — Data Validation** enforces schema correctness, referential integrity, and
domain rules against the ingested data. It also runs Population Stability Index (PSI)
drift detection against a reference baseline captured on the first run, flagging data
distribution shifts before they reach the model. Outputs include `validated_transactions.parquet`,
`validation_status.json`, and `reference_stats.json`.

**pip_03 — Feature Engineering** is the core transformation stage. It fits sender and
device lookup stores on the full training dataset, engineers all 20 features across six
groups (temporal, amount, balance, geographic, velocity, and behavioural), applies
`StandardScaler` to continuous features only, and uses SMOTE to address the class
imbalance (~4% fraud) on the training set. The stage saves `X_train`, `X_test`,
`y_train`, and `y_test` as parquet files alongside `pipeline.joblib` — a serialized
object that contains the fitted transformer, lookup stores, and scaler.

**pip_04 — Model Training** trains a LightGBM classifier on the prepared feature
matrices, with optional Optuna hyperparameter search. All parameters, metrics, and
artifacts are logged to MLflow. Outputs are `lgb_model.joblib` and `run_id.json`.

**pip_05 — Model Evaluation** scores the held-out test set at two decision thresholds
(T=0.30 for high recall, T=0.55 for high precision) and produces a full evaluation
report, confusion matrices, ROC curve, and precision-recall curve — all logged to the
linked MLflow run.

**pip_06 — Batch Prediction** loads `pipeline.joblib` read-only and scores new
transaction batches. Velocity and behavioural features are resolved from the fitted
lookup stores rather than re-aggregated from the batch, ensuring the feature
distribution at inference matches what the model was trained on. The stage also runs
drift monitoring and saves `predictions.csv` and `batch_monitoring_report.json`.

> **Note on the lookup store:** sender and device profiles captured in `pipeline.joblib`
> can be refreshed on a schedule via `pipeline.update_store(recent_transactions_df)`
> without retraining the model, keeping profiles current as transaction behaviour evolves.

---


## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline in order
python src/pipelines/pip_01_data_ingestion.py
python src/pipelines/pip_02_data_validation.py
python src/pipelines/pip_03_feature_engineering.py
python src/pipelines/pip_04_model_training.py
python src/pipelines/pip_05_model_evaluation.py
python src/pipelines/pip_06_batch_prediction.py
```

---
# Fraud Detection Pipeline — Runbook
# =====================================
# Follow this guide every time you need to run the full pipeline
# from scratch, whether for a demo, presentation, or development.
#
# Prerequisites (one-time setup — already done):
#   - Docker installed and running
#   - kind installed
#   - kubectl installed
#   - argo CLI installed
#   - conda environment: financial-fraud (Python 3.10)
# =============================================


# ── SECTION 1: LOCAL PIPELINE (no Docker, no Argo) ────────────────────────
# Use this for quick runs and development

# 1. Activate environment
conda activate financial-fraud

# 2. Navigate to project
cd ~/Desktop/machine-learning/financial-fraud

# 3. Run full pipeline
python main.py

# 4. OR run with DVC (skips unchanged stages)
dvc repro

# 5. OR run individual stages
python src/pipelines/pip_01_data_ingestion.py
python src/pipelines/pip_02_data_validation.py
python src/pipelines/pip_03_feature_engineering.py
python src/pipelines/pip_04_model_training.py
python src/pipelines/pip_05_model_evaluation.py

# 6. Run batch prediction (on demand)
python src/pipelines/pip_06_batch_prediction.py

# 7. View MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000


# ── SECTION 2: DOCKER ─────────────────────────────────────────────────────
# Use this to test the containerized pipeline

# 1. Navigate to project
cd ~/Desktop/machine-learning/financial-fraud

# 2. Build Docker image
docker build -t fraud-detection-pipeline:latest .

# 3. Verify image was created
docker images | grep fraud-detection

# 4. Run full pipeline in Docker
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  fraud-detection-pipeline:latest

# 5. Run a single stage in Docker
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  fraud-detection-pipeline:latest \
  python src/scripts/validate_data.py

# 6. Check running containers
docker ps

# 7. Stop all running containers
docker stop $(docker ps -q)


# ── SECTION 3: ARGO WORKFLOWS (Kubernetes) ────────────────────────────────
# Use this for the full MLOps demo

# --- 3a. START CLUSTER (do this every time after a reboot) ---

# 1. Navigate to project
> cd ~/Desktop/machine-learning/financial-fraud

# 2. Check if kind cluster is already running
> kind get clusters
# Expected output: fraud-detection

# 3a. If cluster EXISTS — just check pods are running
> kubectl get pods -n argo
# Expected: argo-server and workflow-controller both Running

# 3b. If cluster DOES NOT EXIST — create it
> kind create cluster --name fraud-detection --config kind-config.yaml

# 4. Install Argo (only needed if cluster was recreated)
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.10/install.yaml
kubectl wait --for=condition=ready pod --all -n argo --timeout=180s

# 5. Grant permissions (only needed if cluster was recreated)
kubectl create rolebinding argo-default-admin \
  --clusterrole=admin \
  --serviceaccount=argo:default \
  --namespace=argo

# --- 3b. LOAD IMAGE INTO CLUSTER ---
# Required every time you rebuild the Docker image
# OR after recreating the cluster

# 6. Load image into kind
kind load docker-image fraud-detection-pipeline:latest --name fraud-detection

# --- 3c. START ARGO UI ---
# Run this in a DEDICATED terminal — keep it running during the demo

# 7. Port-forward Argo UI (run in separate terminal, do not close)
kubectl -n argo port-forward deployment/argo-server 2746:2746
# Open https://localhost:2746 in browser
# Accept the certificate warning

# --- 3d. SUBMIT WORKFLOW ---
# Run this in your MAIN terminal

# 8. Lint workflow (catch errors before submitting)
argo lint workflow.yaml

# 9. Submit and watch pipeline run
argo submit workflow.yaml --watch -n argo

# 10. List all workflow runs
argo list -n argo

# 11. Get details of a specific run
argo get fraud-detection-pipeline -n argo

# 12. View logs of a specific step
kubectl logs <pod-name> -n argo
# Get pod name from: kubectl get pods -n argo


# ── SECTION 4: FULL RESET (start completely fresh) ────────────────────────
# Use this if something is broken and you want a clean slate

# 1. Delete kind cluster
kind delete cluster --name fraud-detection

# 2. Remove all artifacts (optional — only if you want fresh outputs)
rm -rf artifacts/
mkdir -p artifacts

# 3. Remove MLflow runs (optional)
rm -rf mlruns/
mkdir -p mlruns

# 4. Recreate cluster and reinstall Argo (follow Section 3a steps 3b-5)

# 5. Rebuild and reload Docker image (follow Section 2 steps 2-3, then Section 3b step 6)

# 6. Resubmit workflow (follow Section 3d)

# 7. On restart 
> docker start fraud-detection-control-plane

### check cluster is healthy (wait for both pods to show Running)
> kubectl get pods -n argo

### load image into cluster
> kind load docker-image fraud-detection-pipeline:latest --name fraud-detection

### open a second terminal and run this — keep it open
> kubectl -n argo port-forward deployment/argo-server 2746:2746

### back in main terminal, lint and submit
> argo lint workflow.yaml

### Delete and resubmit 
> argo delete fraud-detection-pipeline -n argo

### Submit 
>> argo submit workflow.yaml --watch -n argo

### Launch 
> https://localhost:2746

# ── SECTION 5: DEMO CHECKLIST ─────────────────────────────────────────────
# Run through this before your presentation

# [ ] conda activate financial-fraud
# [ ] cd ~/Desktop/machine-learning/financial-fraud
# [ ] docker images | grep fraud-detection        ← image exists
# [ ] kind get clusters                            ← fraud-detection cluster exists
# [ ] kubectl get pods -n argo                     ← argo-server and controller Running
# [ ] In separate terminal: kubectl -n argo port-forward deployment/argo-server 2746:2746
# [ ] Open https://localhost:2746                  ← Argo UI accessible
# [ ] argo lint workflow.yaml                      ← no linting errors
# [ ] mlflow ui --port 5000                        ← MLflow UI accessible at http://localhost:5000
# [ ] python main.py                               ← local pipeline runs clean (optional dry run)