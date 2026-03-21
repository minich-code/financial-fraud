
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

## Inference

```python
from src.components.feature_engineering import FraudTransformationPipeline

# Load the fitted pipeline (includes lookup stores + scaler)
pipeline = FraudTransformationPipeline.load("artifacts/feature_engineering/pipeline.joblib")

# Transform a new batch — features resolved from stores, not re-aggregated
X_ready = pipeline.transform(new_batch_df)

# Score
predictions = model.predict(X_ready)
```

To refresh sender and device profiles from recent transaction history:

```python
pipeline.update_store(recent_transactions_df)
pipeline.save("artifacts/feature_engineering/pipeline.joblib")
```

---

## Stack

| Component | Library |
|---|---|
| Model | `lightgbm` |
| Class balancing | `imbalanced-learn` (SMOTE) |
| Experiment tracking | `mlflow` |
| Serialization | `joblib` |
| Data format | `parquet` (via `pandas`) |

