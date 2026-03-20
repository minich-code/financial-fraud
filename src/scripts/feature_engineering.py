# src/feature_engineering.py
# Wrapper script for the feature engineering stage.
# Called directly by Argo Workflow or standalone.
# Usage: python src/feature_engineering.py

import sys
from src.pipelines.pip_03_feature_engineering import FeatureEngineeringPipeline
from src.utils.logger import logger
from src.utils.exception import PipelineException

if __name__ == "__main__":
    try:
        logger.info("Starting feature engineering stage.")
        FeatureEngineeringPipeline().run()
        logger.info("Feature engineering stage complete.")
    except PipelineException as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)