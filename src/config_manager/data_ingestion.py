# config_manager/data_ingestion.py
from dataclasses import dataclass
from pathlib import Path
from src.utils.commons import read_yaml, create_directories
from src.utils.exception import PipelineException, ErrorCategory

DATA_INGESTION_CONFIG_FILEPATH = Path("config/data_ingestion.yaml")

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    root_dir: Path
    source_file: Path
    output_file: str


class ConfigurationManager:
    """Manages configuration loading for pipeline components."""

    def __init__(self, config_filepath: Path = DATA_INGESTION_CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_filepath)
            create_directories([Path(self.config.artifacts_root)])
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Load and return data ingestion configuration."""
        try:
            config = self.config.data_ingestion
            root_dir = Path(config.root_dir)
            create_directories([root_dir])
            return DataIngestionConfig(
                root_dir=root_dir,
                source_file=Path(config.source_file),
                output_file=config.output_file
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.CONFIGURATION)