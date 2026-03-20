# src/train_model.py
# Wrapper script for the model training stage.
# Called directly by Argo Workflow or standalone.
# Usage: python src/train_model.py

import sys
from src.pipelines.pip_04_model_training import ModelTrainingPipeline
from src.utils.logger import logger
from src.utils.exception import PipelineException

if __name__ == "__main__":
    try:
        logger.info("Starting model training stage.")
        ModelTrainingPipeline().run()
        logger.info("Model training stage complete.")
    except PipelineException as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)