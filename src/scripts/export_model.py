# src/export_model.py
# Exports the trained model with metadata for deployment.
# Called directly by Argo Workflow or standalone.
# Usage: python src/export_model.py

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib

from src.utils.logger import logger
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity

# Source artifacts
MODEL_PATH          = Path("artifacts/model_training/lgb_model.joblib")
PIPELINE_PATH       = Path("artifacts/feature_engineering/pipeline.joblib")
EVALUATION_REPORT   = Path("artifacts/model_evaluation/evaluation_report.json")

# Export destination
EXPORT_DIR          = Path("artifacts/model_export")

# Argo artifact output path
ARGO_ARTIFACT_DIR   = Path("/tmp/model_artifact")

# Minimum AUC-ROC required before export is allowed
MIN_AUC_THRESHOLD   = 0.95


def export_model() -> None:
    """
    Validates evaluation metrics, then exports the model and transformation
    pipeline as a versioned artifact bundle ready for deployment.

    Export bundle contains:
      - lgb_model.joblib          — trained LightGBM model
      - pipeline.joblib           — fitted transformation pipeline
      - model_metadata.json       — metrics, thresholds, export timestamp
    """
    try:
        # ── Validate metrics before export ────────────────────────────────────
        if not EVALUATION_REPORT.exists():
            raise FileNotFoundError(
                f"Evaluation report not found: {EVALUATION_REPORT}. "
                "Run evaluate_model.py before exporting."
            )

        with open(EVALUATION_REPORT) as f:
            report = json.load(f)

        auc = report["threshold_independent"]["roc_auc"]
        pr_auc = report["threshold_independent"]["pr_auc"]

        if auc < MIN_AUC_THRESHOLD:
            raise ValueError(
                f"Model AUC-ROC ({auc:.4f}) is below the minimum export "
                f"threshold ({MIN_AUC_THRESHOLD}). Retrain before exporting."
            )

        logger.info(f"Metrics check passed — AUC-ROC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

        # ── Verify source artifacts exist ──────────────────────────────────────
        for path in [MODEL_PATH, PIPELINE_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        # ── Build versioned export bundle ──────────────────────────────────────
        version   = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = EXPORT_DIR / f"v_{version}"
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy model and pipeline
        shutil.copy2(MODEL_PATH,    export_path / "lgb_model.joblib")
        shutil.copy2(PIPELINE_PATH, export_path / "pipeline.joblib")
        logger.info(f"Model and pipeline copied to {export_path}")

        # Build metadata
        metadata = {
            "model_version":    version,
            "export_timestamp": datetime.now().isoformat(),
            "model_type":       "LightGBMClassifier",
            "metrics": {
                "roc_auc": auc,
                "pr_auc":  pr_auc,
                "by_threshold": report.get("by_threshold", {}),
            },
            "thresholds": {
                "suspicious": 0.3,
                "fraud":      0.55,
            },
            "artifacts": {
                "model":    "lgb_model.joblib",
                "pipeline": "pipeline.joblib",
            },
        }

        metadata_path = export_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")

        # ── Write Argo artifact output ─────────────────────────────────────────
        ARGO_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_path / "lgb_model.joblib",     ARGO_ARTIFACT_DIR / "lgb_model.joblib")
        shutil.copy2(export_path / "pipeline.joblib",      ARGO_ARTIFACT_DIR / "pipeline.joblib")
        shutil.copy2(export_path / "model_metadata.json",  ARGO_ARTIFACT_DIR / "model_metadata.json")

        logger.info(
            f"Model exported successfully.\n"
            f"  Version   : {version}\n"
            f"  Location  : {export_path.resolve()}\n"
            f"  AUC-ROC   : {auc:.4f}\n"
            f"  PR-AUC    : {pr_auc:.4f}"
        )

    except (FileNotFoundError, ValueError) as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH
        )
    except PipelineException:
        raise
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH
        )


if __name__ == "__main__":
    try:
        logger.info("Starting model export stage.")
        export_model()
        logger.info("Model export stage complete.")
    except PipelineException as e:
        logger.error(f"Model export failed: {e}")
        sys.exit(1)