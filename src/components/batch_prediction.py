# src/components/batch_prediction.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config_manager.batch_prediction import BatchPredictionConfig
from src.components.feature_engineering import FraudTransformationPipeline
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger


# ── PSI helpers (mirrors data_validation.py) ──────────────────────────────────

def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    return round(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 4)


def _compute_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    all_categories = set(reference.unique()) | set(current.unique())
    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)

    psi = 0.0
    for cat in all_categories:
        r = ref_pct.get(cat, 1e-4)
        c = cur_pct.get(cat, 1e-4)
        psi += (c - r) * np.log(c / r)

    return round(float(psi), 4)


# ── Main component ─────────────────────────────────────────────────────────────

class BatchPredictor:
    """
    Runs batch inference on new transaction data.

    Steps:
      1. Load input CSV and validate required columns are present
      2. Load fitted FraudTransformationPipeline and LightGBM model
      3. Transform batch using the fitted pipeline (no refit)
      4. Generate fraud scores and assign risk tiers
      5. Run drift monitoring against reference_stats.json
      6. Save predictions CSV and monitoring report JSON
      7. Log everything to MLflow (same experiment as training)
    """

    # Columns the transformation pipeline needs — must exist in input
    REQUIRED_COLUMNS = [
        "transaction_id", "timestamp", "sender_id", "receiver_id",
        "amount", "transaction_type", "sender_balance_before",
        "sender_balance_after", "receiver_balance_before",
        "receiver_balance_after", "device_id", "location_lat", "location_lon",
    ]

    NUMERIC_DRIFT_COLS      = ["amount", "sender_balance_before", "receiver_balance_before"]
    CATEGORICAL_DRIFT_COLS  = ["transaction_type"]
    LABEL_DRIFT_COL         = "is_fraud"

    def __init__(self, config: BatchPredictionConfig):
        self.config   = config
        self.run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Path:
        """
        Execute the full batch prediction job.

        Returns:
            Path to the saved predictions CSV.
        """
        try:
            df_raw = self._load_input()
            transformation_pipeline, model = self._load_artifacts()

            df_transformed = self._transform(df_raw, transformation_pipeline)
            df_output      = self._predict(df_raw, df_transformed, model)

            monitoring_report = self._monitor_drift(df_raw)

            predictions_path = self._save_predictions(df_output)
            self._save_monitoring_report(monitoring_report)
            self._log_to_mlflow(df_output, monitoring_report, predictions_path)

            self._log_summary(df_output, monitoring_report)
            return predictions_path

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.MODEL_PREDICTION,
                severity=ErrorSeverity.HIGH
            )

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_input(self) -> pd.DataFrame:
        if not self.config.input_data_path.exists():
            raise FileNotFoundError(
                f"Input data not found: {self.config.input_data_path}"
            )

        df = pd.read_json(self.config.input_data_path)
        logger.info(f"Loaded {len(df):,} transactions from {self.config.input_data_path}")

        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")

        return df

    def _load_artifacts(self) -> Tuple[FraudTransformationPipeline, Any]:
        for path in [self.config.pipeline_path, self.config.model_path]:
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {path}")

        pipeline = FraudTransformationPipeline.load(str(self.config.pipeline_path))
        model    = joblib.load(self.config.model_path)
        logger.info("Transformation pipeline and model loaded successfully.")
        return pipeline, model

    # ── Transform ─────────────────────────────────────────────────────────────

    def _transform(
        self,
        df_raw: pd.DataFrame,
        pipeline: FraudTransformationPipeline,
    ) -> pd.DataFrame:
        """
        Transform raw input using the fitted pipeline.
        scale=True applies the fitted StandardScaler — same as training.
        """
        df_transformed = pipeline.transform(df_raw.copy(), scale=True)
        logger.info(f"Transformation complete — {df_transformed.shape[1]} features.")
        return df_transformed

    # ── Predict ───────────────────────────────────────────────────────────────

    def _predict(
        self,
        df_raw:        pd.DataFrame,
        df_transformed: pd.DataFrame,
        model:         Any,
    ) -> pd.DataFrame:
        """
        Generate fraud scores, apply thresholds, assign risk tiers.
        Carries original id_columns through to the output.
        """
        fraud_scores = model.predict_proba(df_transformed)[:, 1]

        # Assign risk tier based on thresholds
        def _assign_tier(score: float) -> str:
            if score >= self.config.threshold_fraud:
                return self.config.label_fraud
            elif score >= self.config.threshold_suspicious:
                return self.config.label_suspicious
            return self.config.label_legitimate

        risk_tiers = pd.Series(fraud_scores).apply(_assign_tier)

        # Build output — id columns + prediction columns
        id_cols_present = [
            c for c in self.config.id_columns if c in df_raw.columns
        ]
        df_output = df_raw[id_cols_present].copy().reset_index(drop=True)
        df_output["fraud_score"] = fraud_scores.round(4)
        df_output["risk_tier"]   = risk_tiers.values
        df_output["batch_date"]  = self.run_date

        logger.info(
            f"Predictions generated — "
            f"{(risk_tiers == self.config.label_fraud).sum():,} fraud | "
            f"{(risk_tiers == self.config.label_suspicious).sum():,} suspicious | "
            f"{(risk_tiers == self.config.label_legitimate).sum():,} legitimate"
        )
        return df_output

    # ── Drift monitoring ──────────────────────────────────────────────────────

    def _monitor_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute PSI on key columns against the saved reference baseline.
        Mirrors the logic in DataValidation._validate_drift().
        """
        report: Dict[str, Any] = {"drift_detected": False, "psi_scores": {}}

        if not self.config.reference_stats_path.exists():
            logger.warning(
                "reference_stats.json not found — skipping drift monitoring. "
                "Run the validation pipeline first."
            )
            report["note"] = "No reference stats available."
            return report

        with open(self.config.reference_stats_path) as f:
            ref_stats = json.load(f)

        # Numeric PSI
        for col in self.NUMERIC_DRIFT_COLS:
            if col not in df.columns or col not in ref_stats.get("numeric", {}):
                continue
            ref = ref_stats["numeric"][col]
            ref_series = pd.Series(
                np.random.normal(ref["mean"], ref["std"], 1000)
            ).clip(ref["min"], ref["max"])

            psi     = _compute_psi(ref_series, df[col].dropna())
            flagged = psi >= self.config.psi_threshold
            report["psi_scores"][col] = {"psi": psi, "flagged": flagged}

            if flagged:
                logger.warning(f"Drift detected in '{col}': PSI={psi:.4f}")
                report["drift_detected"] = True

        # Categorical PSI — transaction_type
        for col in self.CATEGORICAL_DRIFT_COLS:
            if col not in df.columns or col not in ref_stats.get("categorical", {}):
                continue
            ref_dist   = pd.Series(ref_stats["categorical"][col])
            ref_series = ref_dist.index.repeat(
                (ref_dist.values * 1000).astype(int)
            )
            psi     = _compute_categorical_psi(pd.Series(ref_series), df[col].dropna())
            flagged = psi >= self.config.psi_threshold
            report["psi_scores"][col] = {"psi": psi, "flagged": flagged}

            if flagged:
                logger.warning(f"Drift detected in '{col}': PSI={psi:.4f}")
                report["drift_detected"] = True

        # Fraud rate drift (only if ground truth labels are present)
        if self.LABEL_DRIFT_COL in df.columns and "fraud_rate" in ref_stats:
            current_rate = float(df[self.LABEL_DRIFT_COL].mean())
            ref_rate     = ref_stats["fraud_rate"]
            shift        = abs(current_rate - ref_rate)
            flagged      = shift > self.config.fraud_rate_shift_threshold

            report["psi_scores"][self.LABEL_DRIFT_COL] = {
                "reference_rate": round(ref_rate,     4),
                "current_rate":   round(current_rate, 4),
                "shift":          round(shift,        4),
                "flagged":        flagged,
            }
            if flagged:
                logger.warning(
                    f"Fraud rate shifted: {ref_rate:.4f} → {current_rate:.4f} "
                    f"(delta={shift:.4f})"
                )
                report["drift_detected"] = True

        status = "DRIFT DETECTED — consider retraining" if report["drift_detected"] else "STABLE"
        report["status"] = status
        logger.info(f"Drift monitoring: {status}")
        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_predictions(self, df_output: pd.DataFrame) -> Path:
        path = self.config.root_dir / self.config.predictions_filename
        df_output.to_csv(path, index=False)
        logger.info(f"Predictions saved to {path} ({len(df_output):,} rows)")
        return path

    def _save_monitoring_report(self, report: Dict[str, Any]) -> None:
        path = self.config.root_dir / "batch_monitoring_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=4, default=str)
        logger.info(f"Monitoring report saved to {path}")

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_to_mlflow(
        self,
        df_output:        pd.DataFrame,
        monitoring_report: Dict[str, Any],
        predictions_path: Path,
    ) -> None:
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)

        total       = len(df_output)
        n_fraud     = int((df_output["risk_tier"] == self.config.label_fraud).sum())
        n_suspicious = int((df_output["risk_tier"] == self.config.label_suspicious).sum())
        n_legitimate = int((df_output["risk_tier"] == self.config.label_legitimate).sum())

        with mlflow.start_run(run_name=f"batch_prediction_{self.run_date}"):
            mlflow.set_tags({
                "mode":       "batch_prediction",
                "batch_date": self.run_date,
                "drift":      str(monitoring_report.get("drift_detected", False)),
            })

            # Score distribution metrics
            mlflow.log_metrics({
                "batch_total_transactions": total,
                "batch_fraud_count":        n_fraud,
                "batch_suspicious_count":   n_suspicious,
                "batch_legitimate_count":   n_legitimate,
                "batch_fraud_rate":         round(n_fraud / total, 4),
                "batch_flag_rate":          round((n_fraud + n_suspicious) / total, 4),
                "batch_avg_fraud_score":    round(float(df_output["fraud_score"].mean()), 4),
            })

            # PSI scores
            for col, psi_data in monitoring_report.get("psi_scores", {}).items():
                if "psi" in psi_data:
                    mlflow.log_metric(f"psi_{col}", psi_data["psi"])

            # Artifacts
            mlflow.log_artifact(str(predictions_path), artifact_path="predictions")
            mlflow.log_artifact(
                str(self.config.root_dir / "batch_monitoring_report.json"),
                artifact_path="monitoring"
            )

        logger.info(f"Batch run logged to MLflow experiment '{self.config.experiment_name}'")

    # ── Summary ───────────────────────────────────────────────────────────────

    def _log_summary(
        self,
        df_output:         pd.DataFrame,
        monitoring_report: Dict[str, Any],
    ) -> None:
        total        = len(df_output)
        n_fraud      = int((df_output["risk_tier"] == self.config.label_fraud).sum())
        n_suspicious = int((df_output["risk_tier"] == self.config.label_suspicious).sum())
        n_legitimate = int((df_output["risk_tier"] == self.config.label_legitimate).sum())

        logger.info(
            f"Batch Summary — "
            f"Total: {total:,} | "
            f"Fraud: {n_fraud:,} ({n_fraud/total:.2%}) | "
            f"Suspicious: {n_suspicious:,} ({n_suspicious/total:.2%}) | "
            f"Legitimate: {n_legitimate:,} ({n_legitimate/total:.2%})"
        )
        if monitoring_report.get("drift_detected"):
            logger.warning(
                "ACTION REQUIRED: Drift detected in this batch. "
                "Set run_hyperparameter_search: true in model_training.yaml "
                "and run dvc repro to retrain."
            )