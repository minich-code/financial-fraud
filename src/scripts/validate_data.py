# src/validate_data.py
# Wrapper script for the data validation stage.
# Called directly by Argo Workflow or standalone.
# Usage: python src/validate_data.py

import sys
from src.pipelines.pip_02_data_validation import DataValidationPipeline
from src.utils.logger import logger
from src.utils.exception import PipelineException

if __name__ == "__main__":
    try:
        logger.info("Starting data validation stage.")
        pipeline = DataValidationPipeline()
        status = pipeline.run()
        if not status:
            logger.error("Data validation failed. Check validation_status.json.")
            sys.exit(1)
        logger.info("Data validation stage complete.")
    except PipelineException as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)