# src/pipelines/pip_05_model_evaluation.py
from src.config_manager.model_evaluation import ConfigurationManager
from src.components.model_evaluation import ModelEvaluator
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "MODEL EVALUATION PIPELINE"


class ModelEvaluationPipeline:
    """Orchestrates the model evaluation stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        evaluator = ModelEvaluator(config=model_evaluation_config)
        evaluator.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        ModelEvaluationPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.MODEL_VALIDATION,
            severity=ErrorSeverity.CRITICAL
        )