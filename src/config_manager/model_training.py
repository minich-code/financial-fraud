# src/config_manager/model_training.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

MODEL_TRAINING_CONFIG_FILEPATH = Path("config/model_training.yaml")


@dataclass
class ModelTrainingConfig:
    """Configuration for the model training component."""
    root_dir:                  Path
    X_train_path:              Path
    X_test_path:               Path
    y_train_path:              Path
    y_test_path:               Path
    target_col:                str
    mlflow_uri:                str
    experiment_name:           str
    run_hyperparameter_search: bool
    default_params:            Dict[str, Any]
    optuna:                    Dict[str, Any]


class ConfigurationManager:
    """Manages configuration loading for the model training component."""

    def __init__(self, config_filepath: Path = MODEL_TRAINING_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_model_training_config(self) -> ModelTrainingConfig:
        """Load and return model training configuration."""
        try:
            config = self.config.model_training
            root_dir = Path(config.root_dir)
            create_directories([root_dir])

            return ModelTrainingConfig(
                root_dir=root_dir,
                X_train_path=Path(config.X_train_path),
                X_test_path=Path(config.X_test_path),
                y_train_path=Path(config.y_train_path),
                y_test_path=Path(config.Y_test_path),
                target_col=config.target_col,
                mlflow_uri=config.mlflow_uri,
                experiment_name=config.experiment_name,
                run_hyperparameter_search=config.run_hyperparameter_search,
                default_params=dict(config.default_params),
                optuna=dict(config.optuna),
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)