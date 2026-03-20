# src/scripts/ingest_data.py
import sys
from src.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.utils.logger import logger
from src.utils.exception import PipelineException

if __name__ == "__main__":
    try:
        logger.info("Starting data ingestion stage.")
        DataIngestionPipeline().run()
        logger.info("Data ingestion stage complete.")
    except PipelineException as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)