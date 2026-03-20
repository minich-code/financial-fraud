Raw Data (transactions.csv)
         │
         ▼
┌─────────────────────┐
│   Data Ingestion    │  pip_01 — reads CSV, cleans, saves parquet
└─────────┬───────────┘
          │ transactions.parquet
          ▼
┌─────────────────────┐
│   Data Validation   │  pip_02 — schema, integrity, domain rules, PSI drift
└─────────┬───────────┘
          │ validated_transactions.parquet
          │ validation_status.json
          │ reference_stats.json          ← PSI baseline (first run only)
          ▼
┌─────────────────────┐
│ Feature Engineering │  pip_03 — fit stores, SMOTE, scale, train/test split
└─────────┬───────────┘
          │ X_train / X_test / y_train / y_test (parquet)
          │ pipeline.joblib               ← fitted transformer
          │                                  contains sender/device lookup stores
          │                                  NOTE: stores are refreshed via
          │                                  pipeline.update_store() on a schedule
          │                                  (e.g. daily) using recent transactions
          │                                  so sender profiles stay current
          │                                  without full retraining
          ▼
┌─────────────────────┐
│   Model Training    │  pip_04 — LightGBM, optional Optuna search, MLflow
└─────────┬───────────┘
          │ lgb_model.joblib
          │ run_id.json                   ← links to MLflow run
          ▼
┌─────────────────────┐
│   Model Evaluation  │  pip_05 — metrics at T=0.30 and T=0.55, plots, MLflow
└─────────┬───────────┘
          │ evaluation_report.json
          │ confusion_matrix_*.png
          │ roc_curve.png
          │ precision_recall_curve.png
          ▼
┌─────────────────────┐
│   Batch Prediction  │  pip_06 — score new transactions, drift monitoring
└─────────────────────┘
          │ predictions.csv
          │ batch_monitoring_report.json
          │
          │ NOTE: pipeline.joblib is loaded read-only — no refit on batch.
          │ Sender/device lookup stores resolve features from saved history.
          │ For production real-time scoring, a FastAPI wrapper would load
          │ the same pipeline.joblib and serve single-transaction predictions
          │ via REST API, with the store refreshed from a live transaction DB.