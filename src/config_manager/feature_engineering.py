# src/config_manager/feature_engineering.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

FEATURE_ENGINEERING_CONFIG_FILEPATH = Path("config/feature-engineering.yaml")


@dataclass
class FeatureEngineeringConfig:
    """Configuration for the feature engineering component."""
    root_dir:     Path
    data_path:    Path
    target_col:   str
    test_size:    float
    random_state: int
    smote:        bool
    type_mapping: dict
    nairobi_lat:  float
    nairobi_lon:  float
    drop_cols:    List[str]
    skip_scale_cols: List[str]


class ConfigurationManager:
    """Manages configuration loading for the feature engineering component."""

    def __init__(self, config_filepath: Path = FEATURE_ENGINEERING_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """Load and return feature engineering configuration."""
        try:
            config = self.config.feature_engineering
            root_dir = Path(config.root_dir)
            create_directories([root_dir])

            return FeatureEngineeringConfig(
                root_dir=root_dir,
                data_path=Path(config.data_path),
                target_col=config.target_col,
                test_size=config.test_size,
                random_state=config.random_state,
                smote=config.smote,
                type_mapping=dict(config.type_mapping),
                nairobi_lat=config.nairobi_lat,
                nairobi_lon=config.nairobi_lon,
                drop_cols=list(config.drop_cols),
                skip_scale_cols=list(config.skip_scale_cols),
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)