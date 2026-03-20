import sys
import logging
import traceback
import json
from datetime import datetime
from typing import Any, Dict, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from contextlib import contextmanager

# Type variable for decorator return types
T = TypeVar('T')


class ErrorSeverity(Enum):
    """Enumeration of error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Enumeration of error categories for ML pipeline classification."""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_PREDICTION = "model_prediction"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


# Configure module-level logger
def _configure_logger(name: str = 'FraudDetection') -> logging.Logger:
    """Configure and return a structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# Initialize logger
logger = _configure_logger()


@dataclass(frozen=True)
class ErrorDetails:
    """Immutable dataclass to capture comprehensive error details."""
    exc_type: str
    exc_message: str
    file_name: str
    line_number: int
    function_name: str
    timestamp: str
    stack_trace: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert error details to dictionary format."""
        return asdict(self)


@dataclass
class PipelineException(Exception):
    """
    Custom exception class for the ML Pipeline.
    """

    error: BaseException  # Changed from Exception to BaseException
    error_type: Optional[str] = None
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    message: str = field(init=False)
    error_details: ErrorDetails = field(init=False)

    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        # Set error_type if not provided
        if self.error_type is None:
            object.__setattr__(self, 'error_type', type(self.error).__name__)

        # Capture traceback details
        error_details = self._capture_error_details()
        object.__setattr__(self, 'error_details', error_details)

        # Format the error message
        message = self._format_error_message()
        object.__setattr__(self, 'message', message)

        # Call parent constructor
        super().__init__(self.message)

    def _capture_error_details(self) -> ErrorDetails:
        """Capture comprehensive error details from the current exception context."""
        exc_type, exc_value, exc_tb = sys.exc_info()

        # Default values when traceback is unavailable
        file_name = "unknown"
        line_number = 0
        function_name = "unknown"
        stack_trace = ""

        if exc_tb is not None:
            try:
                # Traverse to the deepest frame
                tb = exc_tb
                while tb.tb_next is not None:
                    tb = tb.tb_next

                frame = tb.tb_frame
                file_name = frame.f_code.co_filename
                line_number = tb.tb_lineno
                function_name = frame.f_code.co_name
                stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            except (AttributeError, TypeError):
                # Fallback if traceback extraction fails
                stack_trace = traceback.format_exc() if exc_type else ""

        return ErrorDetails(
            exc_type=type(self.error).__name__,
            exc_message=str(self.error),
            file_name=file_name,
            line_number=line_number,
            function_name=function_name,
            timestamp=datetime.now().isoformat(),
            stack_trace=stack_trace
        )

    def _format_error_message(self) -> str:
        """Format a detailed, human-readable error message."""
        parts = [
            f"[{self.severity.value.upper()}] {self.error_type}",
            f"Category: {self.category.value}",
            f"Location: {self.error_details.file_name}:{self.error_details.line_number}",
            f"Function: {self.error_details.function_name}",
            f"Message: {self.error_details.exc_message}"
        ]

        if self.error_code:
            parts.insert(1, f"Error Code: {self.error_code}")

        if self.context:
            context_items = [f"    {k}: {v}" for k, v in self.context.items()]
            parts.append(f"Context:\n" + "\n".join(context_items))

        return "\n".join(parts)

    def log_error(self, level: Optional[int] = None) -> None:
        """Log the error with appropriate severity level."""
        if level is None:
            level_map = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }
            level = level_map.get(self.severity, logging.ERROR)

        logger.log(level, self.message)

        # Log stack trace at DEBUG level for troubleshooting
        if self.error_details.stack_trace:
            logger.debug(f"Stack Trace:\n{self.error_details.stack_trace}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "details": self.error_details.to_dict()
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        """Return the formatted error message."""
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"PipelineException(error_type={self.error_type!r}, "
            f"category={self.category.value!r}, "
            f"severity={self.severity.value!r})"
        )


@contextmanager
def error_handler(
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
        log_immediately: bool = True
):
    """Context manager for automatic exception handling."""
    try:
        yield
    except PipelineException:
        # Already a PipelineException, just reraise
        raise
    except Exception as e:
        custom_exc = PipelineException(
            error=e,
            category=category,
            severity=severity,
            context=context or {}
        )
        if log_immediately:
            custom_exc.log_error()
        if reraise:
            raise custom_exc


def handle_exceptions(
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        default_return: Optional[T] = None,
        log_immediately: bool = True
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """Decorator for automatic exception handling in functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except PipelineException:
                raise
            except Exception as e:
                custom_exc = PipelineException(
                    error=e,
                    category=category,
                    severity=severity,
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                if log_immediately:
                    custom_exc.log_error()
                raise custom_exc

        return wrapper

    return decorator


# Convenience functions for common pipeline operations
def raise_data_validation_error(
        message: str,
        context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a data validation error with appropriate category."""
    raise PipelineException(
        error=ValueError(message),
        category=ErrorCategory.DATA_VALIDATION,
        severity=ErrorSeverity.HIGH,
        context=context or {}
    )


def raise_model_training_error(
        message: str,
        context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a model training error with appropriate category."""
    raise PipelineException(
        error=RuntimeError(message),
        category=ErrorCategory.MODEL_TRAINING,
        severity=ErrorSeverity.HIGH,
        context=context or {}
    )


def raise_configuration_error(
        message: str,
        context: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a configuration error with appropriate category."""
    raise PipelineException(
        error=ValueError(message),
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.CRITICAL,
        context=context or {}
    )