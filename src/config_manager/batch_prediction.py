# src/config_manager/batch_prediction.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

BATCH_PREDICTION_CONFIG_FILEPATH = Path("config/batch_prediction.yaml")


@dataclass
class BatchPredictionConfig:
    """Configuration for the batch prediction component."""
    root_dir:                    Path
    input_data_path:             Path
    model_path:                  Path
    pipeline_path:               Path
    reference_stats_path:        Path
    predictions_filename:        str
    mlflow_uri:                  str
    experiment_name:             str
    threshold_suspicious:        float
    threshold_fraud:             float
    label_legitimate:            str
    label_suspicious:            str
    label_fraud:                 str
    psi_threshold:               float
    fraud_rate_shift_threshold:  float
    id_columns:                  List[str]


class ConfigurationManager:
    """Manages configuration loading for the batch prediction component."""

    def __init__(self, config_filepath: Path = BATCH_PREDICTION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_batch_prediction_config(self) -> BatchPredictionConfig:
        """Load and return batch prediction configuration."""
        try:
            config = self.config.batch_prediction
            root_dir = Path(config.root_dir)
            create_directories([root_dir])

            return BatchPredictionConfig(
                root_dir=root_dir,
                input_data_path=Path(config.input_data_path),
                model_path=Path(config.model_path),
                pipeline_path=Path(config.pipeline_path),
                reference_stats_path=Path(config.reference_stats_path),
                predictions_filename=config.predictions_filename,
                mlflow_uri=config.mlflow_uri,
                experiment_name=config.experiment_name,
                threshold_suspicious=float(config.threshold_suspicious),
                threshold_fraud=float(config.threshold_fraud),
                label_legitimate=config.label_legitimate,
                label_suspicious=config.label_suspicious,
                label_fraud=config.label_fraud,
                psi_threshold=float(config.psi_threshold),
                fraud_rate_shift_threshold=float(config.fraud_rate_shift_threshold),
                id_columns=list(config.id_columns),
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)