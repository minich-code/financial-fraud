# src/pipelines/pip_06_batch_prediction.py
from src.config_manager.batch_prediction import ConfigurationManager
from src.components.batch_prediction import BatchPredictor
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "BATCH PREDICTION PIPELINE"


class BatchPredictionPipeline:
    """Orchestrates the batch prediction stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        batch_prediction_config = config_manager.get_batch_prediction_config()
        predictor = BatchPredictor(config=batch_prediction_config)
        predictions_path = predictor.run()
        logger.info(f"Predictions available at: {predictions_path}")


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        BatchPredictionPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.MODEL_PREDICTION,
            severity=ErrorSeverity.CRITICAL
        )