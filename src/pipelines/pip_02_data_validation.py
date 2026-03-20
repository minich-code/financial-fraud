# src/pipelines/pip_02_data_validation.py
from src.config_manager.data_validation import ConfigurationManager
from src.components.data_validation import DataValidation
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "DATA VALIDATION PIPELINE"


class DataValidationPipeline:
    """Orchestrates the data validation stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        status = data_validation.validate()

        if status:
            logger.info("Data validation passed. Validated parquet saved.")
        else:
            logger.error("Data validation failed. Check validation_status.json for details.")

        return status


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        DataValidationPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.CRITICAL
        )