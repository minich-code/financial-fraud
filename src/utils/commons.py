import os
import sys
import json
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional
from ensure import ensure_annotations
import box
from src.utils.logger import logger
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity


def read_yaml(path_to_yaml: Path) -> box.ConfigBox:
    """Read YAML file and return as ConfigBox for attribute-style access."""
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file loaded: {path_to_yaml}")
            return box.ConfigBox(content)
    except FileNotFoundError:
        raise PipelineException(
            error=FileNotFoundError(f"YAML file not found: {path_to_yaml}"),
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH
        )
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH
        )


def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """Create directories if they don't exist."""
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory: {path}")
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM
        )



def save_object(obj: Any, file_path: Path) -> None:
    """Save Python object using joblib (models, scalers, encoders, etc.)."""
    try:
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved: {file_path}")
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH
        )


def load_object(file_path: Path) -> Any:
    """Load Python object from joblib file."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded: {file_path}")
        return obj
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH
        )


def save_json(data: dict, path: Path) -> None:
    """Save dictionary as JSON file."""
    try:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM
        )


def load_json(path: Path) -> box.ConfigBox:
    """Load JSON file as ConfigBox."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON loaded: {path}")
        return box.ConfigBox(content)
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH
        )


def get_size(path: Path) -> str:
    """Get file size in KB."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        size_kb = round(os.path.getsize(path) / 1024)
        return f"{size_kb} KB"
    except Exception as e:
        raise PipelineException(
            error=e,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.LOW
        )


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp string for versioning artifacts."""
    return datetime.now().strftime(format)


def get_latest_file(directory: Path, pattern: str = "*") -> Optional[Path]:
    """Get the most recently modified file matching pattern in directory."""
    try:
        directory = Path(directory)
        if not directory.exists():
            return None
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    except Exception as e:
        logger.warning(f"Error finding latest file: {e}")
        return None