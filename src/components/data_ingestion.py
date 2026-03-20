# src/components/data_ingestion.py
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_manager.data_ingestion import DataIngestionConfig
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger


class DataIngestion:
    """Handles data ingestion from source to parquet format."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self) -> Path:
        """
        Main ingestion method - reads, cleans, and saves data.

        Returns:
            Path to the saved parquet file

        Raises:
            PipelineException: If ingestion fails at any stage
        """
        try:
            df = self._read_data()

            if df.empty:
                raise PipelineException(
                    error=ValueError("Source file contains no data"),
                    category=ErrorCategory.DATA_INGESTION,
                    severity=ErrorSeverity.HIGH
                )

            cleaned_df = self._clean_data(df)
            return self._save_data(cleaned_df)

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.DATA_INGESTION)

    def _read_data(self) -> pd.DataFrame:
        """Read data from source CSV file."""
        try:
            source_path = self.config.source_file

            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            df = pd.read_csv(source_path)
            logger.info(f"Loaded {len(df)} records from {source_path}")
            return df

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.DATA_INGESTION)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the raw data."""
        try:
            initial_rows = len(df)

            # Drop constant columns (single unique value)
            nunique = df.nunique()
            constant_cols = nunique[nunique == 1].index.tolist()
            if constant_cols:
                df = df.drop(columns=constant_cols)
                logger.info(f"Dropped constant columns: {constant_cols}")

            # Drop zero-variance numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            zero_var_cols = [col for col in numeric_cols if df[col].var() == 0]
            if zero_var_cols:
                df = df.drop(columns=zero_var_cols)
                logger.info(f"Dropped zero-variance columns: {zero_var_cols}")

            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)

            # Coerce numeric columns and drop rows with any remaining NaNs
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()

            final_rows = len(df)
            logger.info(
                f"Data cleaning complete: {initial_rows} -> {final_rows} rows "
                f"({initial_rows - final_rows} rows removed)"
            )
            return df

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.DATA_TRANSFORMATION)

    def _save_data(self, df: pd.DataFrame) -> Path:
        """Save cleaned data to parquet format."""
        try:
            output_path = self.config.root_dir / self.config.output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return output_path

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(e, category=ErrorCategory.DATA_INGESTION)