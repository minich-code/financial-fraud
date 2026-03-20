# src/pipelines/pip_03_feature_engineering.py
from src.config_manager.feature_engineering import ConfigurationManager
from src.components.feature_engineering import DataTransformation
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "FEATURE ENGINEERING PIPELINE"


class FeatureEngineeringPipeline:
    """Orchestrates the feature engineering stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        feature_engineering_config = config_manager.get_feature_engineering_config()
        data_transformation = DataTransformation(config=feature_engineering_config)
        data_transformation.run()


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        FeatureEngineeringPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.DATA_TRANSFORMATION,
            severity=ErrorSeverity.CRITICAL
        )