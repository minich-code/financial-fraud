# src/pipelines/pip_04_model_training.py
from src.config_manager.model_training import ConfigurationManager
from src.components.model_training import ModelTrainer
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "MODEL TRAINING PIPELINE"


class ModelTrainingPipeline:
    """Orchestrates the model training stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        trainer = ModelTrainer(config=model_training_config)
        trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        ModelTrainingPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.MODEL_TRAINING,
            severity=ErrorSeverity.CRITICAL
        )