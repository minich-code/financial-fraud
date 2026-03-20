# src/evaluate_model.py
# Wrapper script for the model evaluation stage.
# Called directly by Argo Workflow or standalone.
# Usage: python src/evaluate_model.py

import json
import sys
from pathlib import Path

from src.pipelines.pip_05_model_evaluation import ModelEvaluationPipeline
from src.utils.logger import logger
from src.utils.exception import PipelineException

# Paths Argo reads as output parameters
AUC_SCORE_PATH = Path("/tmp/auc_score.txt")
METRICS_PATH   = Path("/tmp/metrics.json")


def _write_argo_outputs() -> None:
    """
    Write evaluation outputs to /tmp for Argo artifact/parameter passing.
    Reads from the evaluation report saved by the pipeline.
    """
    report_path = Path("artifacts/model_evaluation/evaluation_report.json")
    if not report_path.exists():
        logger.warning("Evaluation report not found — skipping Argo output writing.")
        return

    with open(report_path) as f:
        report = json.load(f)

    # Write AUC score as plain text for Argo parameter extraction
    auc = report["threshold_independent"]["roc_auc"]
    AUC_SCORE_PATH.write_text(str(auc))

    # Write full metrics for Argo artifact passing
    METRICS_PATH.write_text(json.dumps(report, indent=4))
    logger.info(f"Argo outputs written — AUC: {auc}")


if __name__ == "__main__":
    try:
        logger.info("Starting model evaluation stage.")
        ModelEvaluationPipeline().run()
        _write_argo_outputs()
        logger.info("Model evaluation stage complete.")
    except PipelineException as e:
        logger.error(f"Model evaluation failed: {e}")
        sys.exit(1)