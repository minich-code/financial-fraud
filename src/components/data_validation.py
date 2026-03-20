# src/components/data_validation.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_manager.data_validation import DataValidationConfig
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index between a reference and current series.
    PSI < 0.1  → no significant shift
    PSI < 0.2  → moderate shift, worth monitoring
    PSI >= 0.2 → significant shift, investigate
    """
    # Build bin edges from the reference distribution
    breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Replace zeros to avoid division-by-zero / log(0)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def _compute_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for categorical columns — uses category proportions as bins."""
    all_categories = set(reference.unique()) | set(current.unique())

    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)

    psi = 0.0
    for cat in all_categories:
        r = ref_pct.get(cat, 1e-4)
        c = cur_pct.get(cat, 1e-4)
        psi += (c - r) * np.log(c / r)

    return round(float(psi), 4)


# ---------------------------------------------------------------------------
# Main component
# ---------------------------------------------------------------------------

class DataValidation:
    """
    Validates ingested transaction data across four dimensions:
      1. Schema validation   — columns present, correct dtypes
      2. Data integrity      — duplicates, null rows
      3. Domain rules        — business constraints (positive amounts, valid coords, etc.)
      4. Statistical drift   — PSI on key columns against a saved reference baseline
    """

    # Columns used for drift tracking
    NUMERIC_DRIFT_COLS = ["amount", "sender_balance_before", "receiver_balance_before"]
    CATEGORICAL_DRIFT_COLS = ["transaction_type"]
    LABEL_DRIFT_COL = "is_fraud"

    # Dtype groups accepted per schema type
    DTYPE_MAPPING = {
        "string":   ["object", "category", "string"],
        "integer":  ["int32", "int64"],
        "float":    ["float32", "float64"],
        "number":   ["float32", "float64"],
        "boolean":  ["bool"],
        "datetime": ["datetime64[ns]", "datetime64"],
    }

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.validation_results: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Run all validation checks. Saves a status JSON and, on success,
        persists the validated parquet.

        Returns:
            True if all checks passed, False otherwise.
        """
        try:
            df = self._load_data()

            # --- run all checks, collect results ---
            schema_ok   = self._validate_schema(df)
            integrity_ok = self._validate_integrity(df)
            domain_ok   = self._validate_domain_rules(df)
            drift_ok    = self._validate_drift(df)

            overall = all([schema_ok, integrity_ok, domain_ok, drift_ok])
            self.validation_results["overall_status"] = "PASSED" if overall else "FAILED"

            self._save_validation_status()

            if overall:
                self._save_validated_data(df)

            return overall

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.DATA_VALIDATION,
                severity=ErrorSeverity.HIGH
            )

    # ------------------------------------------------------------------
    # 1. Schema validation
    # ------------------------------------------------------------------

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        logger.info("Running schema validation...")
        results = {}
        passed = True
        schema = dict(self.config.schema)

        # Check for missing columns
        missing = [col for col in schema if col not in df.columns]
        if missing:
            results["missing_columns"] = missing
            logger.error(f"Missing columns: {missing}")
            passed = False

        # Check for unexpected extra columns
        extra = [col for col in df.columns if col not in schema]
        if extra:
            results["unexpected_columns"] = extra
            logger.warning(f"Unexpected columns not in schema: {extra}")

        # Dtype checks on columns that are present
        dtype_errors = {}
        for col, spec in schema.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            expected_type = spec.get("type") if isinstance(spec, dict) else spec["type"]
            allowed = self.DTYPE_MAPPING.get(expected_type, [])
            if actual not in allowed:
                dtype_errors[col] = f"expected {expected_type}, got {actual}"
                passed = False

        if dtype_errors:
            results["dtype_errors"] = dtype_errors
            logger.error(f"Dtype mismatches: {dtype_errors}")

        results["status"] = "PASSED" if passed else "FAILED"
        self.validation_results["schema_validation"] = results
        logger.info(f"Schema validation: {results['status']}")
        return passed

    # ------------------------------------------------------------------
    # 2. Data integrity
    # ------------------------------------------------------------------

    def _validate_integrity(self, df: pd.DataFrame) -> bool:
        logger.info("Running data integrity checks...")
        results = {}
        passed = True

        # Duplicates
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows — dropping them.")
            df.drop_duplicates(inplace=True)
        results["duplicates_removed"] = duplicate_count

        # Null values — record per column then drop
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        if null_cols:
            logger.warning(f"Null values found, dropping affected rows: {null_cols}")
            df.dropna(inplace=True)
        results["null_columns_dropped"] = {k: int(v) for k, v in null_cols.items()}

        # Row count sanity check
        results["row_count_after_cleaning"] = len(df)
        if len(df) == 0:
            results["status"] = "FAILED"
            results["error"] = "No rows remaining after cleaning"
            logger.error("Data integrity check failed: dataset is empty after cleaning")
            passed = False
        else:
            results["status"] = "PASSED"

        self.validation_results["data_integrity"] = results
        logger.info(f"Data integrity: {results['status']}")
        return passed

    # ------------------------------------------------------------------
    # 3. Domain-specific rules
    # ------------------------------------------------------------------

    def _validate_domain_rules(self, df: pd.DataFrame) -> bool:
        logger.info("Running domain-specific rule checks...")
        results = {}
        passed = True
        schema = dict(self.config.schema)

        for col, spec in schema.items():
            if col not in df.columns:
                continue

            spec = dict(spec)
            constraints = spec.get("constraints", {})
            if not constraints:
                continue

            col_errors = []

            # Enum check
            if "enum" in constraints:
                allowed = set(constraints["enum"])
                invalid = set(df[col].unique()) - allowed
                if invalid:
                    col_errors.append(f"Invalid values: {invalid}")
                    passed = False

            # Range check
            if "minimum" in constraints:
                violations = int((df[col] < constraints["minimum"]).sum())
                if violations:
                    col_errors.append(f"{violations} rows below minimum {constraints['minimum']}")
                    passed = False

            if "maximum" in constraints:
                violations = int((df[col] > constraints["maximum"]).sum())
                if violations:
                    col_errors.append(f"{violations} rows above maximum {constraints['maximum']}")
                    passed = False

            if col_errors:
                results[col] = col_errors
                logger.error(f"Domain rule violations in '{col}': {col_errors}")

        results["status"] = "PASSED" if passed else "FAILED"
        self.validation_results["domain_rules"] = results
        logger.info(f"Domain rules: {results['status']}")
        return passed

    # ------------------------------------------------------------------
    # 4. Statistical drift (PSI)
    # ------------------------------------------------------------------

    def _validate_drift(self, df: pd.DataFrame) -> bool:
        """
        On first run: split 80/20, save reference stats from the 80% slice.
        On subsequent runs: load saved reference stats and compare against
        the full incoming dataset.
        PSI thresholds: <0.1 stable | 0.1–0.2 monitor | >=0.2 flag
        """
        logger.info("Running statistical drift checks...")
        results = {}
        passed = True
        psi_threshold = 0.2

        reference_path = self.config.reference_stats

        if not reference_path.exists():
            # --- First run: build and save reference from 80% slice ---
            logger.info("No reference stats found. Creating baseline from 80% split.")
            reference_df = df.sample(frac=0.8, random_state=42)
            self._save_reference_stats(reference_df)
            results["note"] = "First run — reference baseline created from 80% split. PSI checks will run from next run."
            results["status"] = "PASSED"
            self.validation_results["drift_analysis"] = results
            return True

        # --- Subsequent runs: load reference and compute PSI ---
        with open(reference_path) as f:
            ref_stats = json.load(f)

        psi_results = {}

        # Numeric PSI
        for col in self.NUMERIC_DRIFT_COLS:
            if col not in df.columns or col not in ref_stats.get("numeric", {}):
                continue
            ref = ref_stats["numeric"][col]
            ref_series = pd.Series(
                np.random.normal(ref["mean"], ref["std"], 1000)
            ).clip(ref["min"], ref["max"])
            psi = _compute_psi(ref_series, df[col].dropna())
            flag = psi >= psi_threshold
            psi_results[col] = {"psi": psi, "flagged": flag}
            if flag:
                logger.warning(f"Drift detected in '{col}': PSI={psi}")
                passed = False

        # Categorical PSI — transaction_type
        for col in self.CATEGORICAL_DRIFT_COLS:
            if col not in df.columns or col not in ref_stats.get("categorical", {}):
                continue
            ref_dist = pd.Series(ref_stats["categorical"][col])
            # Reconstruct a reference series from saved proportions
            ref_series = ref_dist.index.repeat(
                (ref_dist.values * 1000).astype(int)
            )
            psi = _compute_categorical_psi(pd.Series(ref_series), df[col].dropna())
            flag = psi >= psi_threshold
            psi_results[col] = {"psi": psi, "flagged": flag}
            if flag:
                logger.warning(f"Drift detected in '{col}': PSI={psi}")
                passed = False

        # Fraud rate drift — simple absolute shift check
        if self.LABEL_DRIFT_COL in df.columns and "fraud_rate" in ref_stats:
            current_rate = float(df[self.LABEL_DRIFT_COL].mean())
            ref_rate = ref_stats["fraud_rate"]
            shift = abs(current_rate - ref_rate)
            flagged = shift > 0.05  # flag if fraud rate shifts more than 5 percentage points
            psi_results[self.LABEL_DRIFT_COL] = {
                "reference_rate": round(ref_rate, 4),
                "current_rate": round(current_rate, 4),
                "shift": round(shift, 4),
                "flagged": flagged
            }
            if flagged:
                logger.warning(
                    f"Fraud rate shifted from {ref_rate:.4f} to {current_rate:.4f} "
                    f"(delta={shift:.4f})"
                )
                passed = False

        results["psi_scores"] = psi_results
        results["status"] = "PASSED" if passed else "FLAGGED"
        self.validation_results["drift_analysis"] = results
        logger.info(f"Drift analysis: {results['status']}")

        # Drift flags don't fail the pipeline — they're warnings
        return True

    def _save_reference_stats(self, df: pd.DataFrame) -> None:
        """Persist summary statistics for drift reference."""
        ref_stats = {"numeric": {}, "categorical": {}}

        for col in self.NUMERIC_DRIFT_COLS:
            if col in df.columns:
                ref_stats["numeric"][col] = {
                    "mean": float(df[col].mean()),
                    "std":  float(df[col].std()),
                    "min":  float(df[col].min()),
                    "max":  float(df[col].max()),
                }

        for col in self.CATEGORICAL_DRIFT_COLS:
            if col in df.columns:
                ref_stats["categorical"][col] = (
                    df[col].value_counts(normalize=True).to_dict()
                )

        if self.LABEL_DRIFT_COL in df.columns:
            ref_stats["fraud_rate"] = float(df[self.LABEL_DRIFT_COL].mean())

        self.config.reference_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.reference_stats, "w") as f:
            json.dump(ref_stats, f, indent=4)

        logger.info(f"Reference stats saved to {self.config.reference_stats}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        try:
            if not self.config.data_dir.exists():
                raise FileNotFoundError(f"Input file not found: {self.config.data_dir}")
            df = pd.read_parquet(self.config.data_dir)
            logger.info(f"Loaded {len(df)} records from {self.config.data_dir}")
            return df
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.DATA_VALIDATION,
                severity=ErrorSeverity.HIGH
            )

    def _save_validation_status(self) -> None:
        try:
            self.config.val_status.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.val_status, "w") as f:
                json.dump(self.validation_results, f, indent=4, default=str)
            logger.info(f"Validation status saved to {self.config.val_status}")
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.DATA_VALIDATION,
                severity=ErrorSeverity.MEDIUM
            )

    def _save_validated_data(self, df: pd.DataFrame) -> None:
        try:
            self.config.validated_data.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.config.validated_data, index=False)
            logger.info(f"Validated data saved to {self.config.validated_data}")
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.DATA_VALIDATION,
                severity=ErrorSeverity.HIGH
            )