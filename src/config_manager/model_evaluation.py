# src/config_manager/model_evaluation.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

MODEL_EVALUATION_CONFIG_FILEPATH = Path("config/model_evaluation.yaml")


@dataclass
class ModelEvaluationConfig:
    """Configuration for the model evaluation component."""
    root_dir:         Path
    model_path:       Path
    X_test_path:      Path
    y_test_path:      Path
    run_id_path:      Path
    mlflow_uri:       str
    experiment_name:  str
    thresholds:       List[float]
    plot_dpi:         int
    plot_style:       str
    color_primary:    str
    color_secondary:  str
    color_tertiary:   str
    color_diagonal:   str


class ConfigurationManager:
    """Manages configuration loading for the model evaluation component."""

    def __init__(self, config_filepath: Path = MODEL_EVALUATION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Load and return model evaluation configuration."""
        try:
            config = self.config.model_evaluation
            root_dir = Path(config.root_dir)
            create_directories([root_dir])

            return ModelEvaluationConfig(
                root_dir=root_dir,
                model_path=Path(config.model_path),
                X_test_path=Path(config.X_test_path),
                y_test_path=Path(config.y_test_path),
                run_id_path=Path(config.run_id_path),
                mlflow_uri=config.mlflow_uri,
                experiment_name=config.experiment_name,
                thresholds=list(config.thresholds),
                plot_dpi=int(config.plot_dpi),
                plot_style=config.plot_style,
                color_primary=config.color_primary,
                color_secondary=config.color_secondary,
                color_tertiary=config.color_tertiary,
                color_diagonal=config.color_diagonal,
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)