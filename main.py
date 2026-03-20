# main.py
# Single entry point — runs the full fraud detection ML pipeline in sequence.
# Usage: python main.py

import sys
from src.utils.logger import logger
from src.utils.exception import PipelineException

from src.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.pipelines.pip_02_data_validation import DataValidationPipeline
from src.pipelines.pip_03_feature_engineering import FeatureEngineeringPipeline
from src.pipelines.pip_04_model_training import ModelTrainingPipeline
from src.pipelines.pip_05_model_evaluation import ModelEvaluationPipeline


STAGES = [
    ("Data Ingestion",       DataIngestionPipeline),
    ("Data Validation",      DataValidationPipeline),
    ("Feature Engineering",  FeatureEngineeringPipeline),
    ("Model Training",       ModelTrainingPipeline),
    ("Model Evaluation",     ModelEvaluationPipeline),
]


def main():
    for stage_name, PipelineClass in STAGES:
        logger.info(f"{'='*20} Starting: {stage_name} {'='*20}")
        try:
            PipelineClass().run()
            logger.info(f"{'='*20} Completed: {stage_name} {'='*20}\n")
        except PipelineException as e:
            logger.error(f"Pipeline failed at stage '{stage_name}': {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error at stage '{stage_name}': {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()