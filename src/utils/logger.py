import logging
import logging.config
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger

# Check if JSON logger is available
HAS_JSON_LOGGER = True

# Constants
FORMATTER_JSON = 'json'
FORMATTER_DETAILED = 'detailed'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 20
DEFAULT_LOGGER_NAME = 'FraudDetection'


class LoggerConfigurator:
    """
    Configures and manages application logging with file and console outputs.
    Supports JSON logging for files and detailed formatting for console.
    """

    _instance: Optional['LoggerConfigurator'] = None
    _configured: bool = False

    def __new__(cls, *args, **kwargs) -> 'LoggerConfigurator':
        """Singleton pattern to ensure single logger configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            log_dir: Optional[str] = None,
            log_file_name: Optional[str] = None,
            max_bytes: int = DEFAULT_MAX_BYTES,
            backup_count: int = DEFAULT_BACKUP_COUNT,
            log_level: str = 'INFO',
            enable_file_logging: bool = True,
            enable_console_logging: bool = True,
    ):
        """
        Initialize the logger configurator.
        """
        # Skip re-initialization if already configured
        if self._configured:
            return

        self.log_dir = log_dir or os.getenv('LOG_DIR', os.getcwd())
        self.log_file_name = log_file_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = self._validate_log_level(log_level)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging

        self._configure_logging()
        self._configured = True

    def _validate_log_level(self, level: str) -> str:
        """Validate and return the log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level_upper = level.upper()
        if level_upper not in valid_levels:
            print(f"Invalid log level '{level}', defaulting to 'INFO'")
            return 'INFO'
        return level_upper

    def _get_formatters(self) -> Dict[str, Dict[str, Any]]:
        """Define formatters for different logging outputs."""
        formatters = {
            'standard': {
                'format': '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            FORMATTER_DETAILED: {
                'format': '[%(asctime)s] [%(levelname)-8s] %(name)s - %(module)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        }

        # Add JSON formatter if available
        if HAS_JSON_LOGGER:
            formatters[FORMATTER_JSON] = {
                '()': jsonlogger.JsonFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(lineno)d %(message)s'
            }

        return formatters

    def _get_log_file_path(self) -> str:
        """Generate and ensure the log file path exists."""
        if self.log_file_name:
            log_file_name = self.log_file_name
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            log_file_name = f"fraud_detection_{timestamp}.log"

        log_path = Path(self.log_dir) / 'logs' / log_file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    def _get_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Define handlers for file and console logging."""
        handlers = {}

        if self.enable_file_logging:
            formatter = FORMATTER_JSON if HAS_JSON_LOGGER else FORMATTER_DETAILED
            handlers['file'] = {
                'level': self.log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': self._get_log_file_path(),
                'maxBytes': self.max_bytes,
                'backupCount': self.backup_count,
                'encoding': 'utf8',
                'formatter': formatter,
                'delay': True  # Delay file opening until first log
            }

        if self.enable_console_logging:
            handlers['console'] = {
                'level': self.log_level,
                'class': 'logging.StreamHandler',
                'formatter': FORMATTER_DETAILED,
                'stream': sys.stdout
            }

        return handlers

    def _configure_logging(self) -> None:
        """Apply the logging configuration."""
        try:
            handlers = self._get_handlers()
            handler_names = list(handlers.keys())

            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': self._get_formatters(),
                'handlers': handlers,
                'root': {  # Configure root logger instead of specific logger
                    'handlers': handler_names,
                    'level': self.log_level,
                },
                'loggers': {
                    DEFAULT_LOGGER_NAME: {
                        'handlers': handler_names,
                        'level': self.log_level,
                        'propagate': False
                    },
                    # Suppress verbose logs from external libraries
                    'urllib3': {
                        'level': 'WARNING',
                        'propagate': True
                    },
                    'matplotlib': {
                        'level': 'WARNING',
                        'propagate': True
                    },
                },
            }
            logging.config.dictConfig(logging_config)
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            logging.warning(f"Failed to configure custom logging: {e}. Using basic logging.")

    def get_logger(self, name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
        """
        Get a configured logger instance.
        """
        return logging.getLogger(name)

    def shutdown(self) -> None:
        """Clean shutdown of logging system."""
        logging.shutdown()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        # Remove all handlers from existing loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

        # Clear root logger handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        cls._instance = None
        cls._configured = False


# Create and configure the logger (singleton)
logger_configurator = LoggerConfigurator()
logger = logger_configurator.get_logger()


# Convenience function for getting child loggers
def get_logger(module_name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    """
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{module_name}")


# Decorator for logging function execution
def log_function_call(logger_instance: Optional[logging.Logger] = None):
    """
    Decorator to log function entry, exit, and exceptions.
    """

    def decorator(func):
        nonlocal logger_instance
        if logger_instance is None:
            logger_instance = logger

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger_instance.debug(f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger_instance.debug(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger_instance.exception(f"Exception in {func_name}: {str(e)}")
                raise

        return wrapper

    return decorator

