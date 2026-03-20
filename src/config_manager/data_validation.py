# src/config_manager/data_validation.py
from dataclasses import dataclass
from pathlib import Path
from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

DATA_VALIDATION_CONFIG_FILEPATH = Path("config/data-validation.yaml")


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    root_dir: Path
    data_dir: Path
    val_status: Path
    validated_data: Path
    reference_stats: Path
    schema: dict


class ConfigurationManager:
    """Manages configuration loading for the data validation component."""

    def __init__(self, config_filepath: Path = DATA_VALIDATION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_data_validation_config(self) -> DataValidationConfig:
        """Load and return data validation configuration."""
        try:
            config = self.config.data_validation
            root_dir = Path(config.root_dir)
            create_directories([root_dir])

            schema = read_yaml(Path(config.schema_path))

            return DataValidationConfig(
                root_dir=root_dir,
                data_dir=Path(config.data_dir),
                val_status=Path(config.val_status),
                validated_data=Path(config.validated_data),
                reference_stats=Path(config.reference_stats),
                schema=schema,
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)