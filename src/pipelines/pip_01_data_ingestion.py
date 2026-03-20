# src/pipeline/pip_01_data_ingestion.py
from src.config_manager.data_ingestion import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

PIPELINE_NAME = "DATA INGESTION PIPELINE"


class DataIngestionPipeline:
    """Orchestrates the data ingestion stage of the ML pipeline."""

    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        output_path = data_ingestion.ingest()
        logger.info(f"Data ingestion complete. Artifact saved to: {output_path}")


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} Starting {PIPELINE_NAME} {'='*20}")
        DataIngestionPipeline().run()
        logger.info(f"{'='*20} {PIPELINE_NAME} Completed Successfully {'='*20}")
    except PipelineException as e:
        logger.error(f"{PIPELINE_NAME} failed: {e}")
        raise
    except Exception as e:
        logger.error(f"{PIPELINE_NAME} failed with unexpected error: {e}")
        raise PipelineException(
            error=e,
            category=ErrorCategory.DATA_INGESTION,
            severity=ErrorSeverity.CRITICAL
        )