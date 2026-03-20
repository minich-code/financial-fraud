# src/components/feature_engineering.py
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_manager.feature_engineering import FeatureEngineeringConfig
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger


class FraudTransformationPipeline:
    """
    Two-phase pipeline:

    fit(df_train)
        Learns all lookup stores and the StandardScaler from training data.
        Must be called once before transform().

    transform(df)
        Applies every feature-engineering step using the fitted stores.
        Safe to call on a single row or a large batch — no groupby on df.

    update_store(new_transactions_df)
        Recomputes sender/device stores from fresh transaction history.
        Call this on a schedule (daily / weekly) so profiles stay current.

    save(path) / load(path)
        Serialize / deserialize the entire fitted pipeline with joblib.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

        # Geographic reference — driven by config, not hardcoded
        self._NAIROBI_LAT: float = config.nairobi_lat
        self._NAIROBI_LON: float = config.nairobi_lon

        # Lookup stores — populated by fit() or update_store()
        self._sender_store: dict = {}
        self._device_store: dict = {}

        # Fallback values for unseen senders / devices (median of training set)
        self._fallbacks: dict = {}

        # Scaler — fitted on continuous training features only
        self._scaler: Optional[StandardScaler] = None
        self._scale_cols: list = []   # populated by fit_scaler()

        self._is_fitted: bool = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> "FraudTransformationPipeline":
        """
        Build sender and device lookup stores from the full training DataFrame.
        Also computes fallback medians for unseen entities at inference time.
        """
        df = df_train.copy()

        sender_agg = df.groupby("sender_id").agg(
            total_tx       =("transaction_id", "count"),
            unique_recv    =("receiver_id",    "nunique"),
            unique_devices =("device_id",      "nunique"),
            avg_amount     =("amount",         "mean"),
        )

        primary_device = (
            df.groupby(["sender_id", "device_id"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .drop_duplicates("sender_id")
            .set_index("sender_id")["device_id"]
            .rename("primary_device")
        )
        sender_agg = sender_agg.join(primary_device)
        self._sender_store = sender_agg.to_dict(orient="index")

        device_agg = df.groupby("device_id").agg(
            unique_senders=("sender_id", "nunique"),
        )
        self._device_store = device_agg.to_dict(orient="index")

        self._fallbacks = {
            "total_tx":         int(sender_agg["total_tx"].median()),
            "unique_recv":      int(sender_agg["unique_recv"].median()),
            "unique_devices":   int(sender_agg["unique_devices"].median()),
            "avg_amount":       float(sender_agg["avg_amount"].median()),
            "is_device_switch": 0,
            "unique_senders":   1,
        }

        self._is_fitted = True
        return self

    # ── fit_scaler ────────────────────────────────────────────────────────────

    def fit_scaler(self, X_train: pd.DataFrame) -> "FraudTransformationPipeline":
        """
        Fit StandardScaler on continuous features only.
        Binary flags are intentionally excluded — scaling 0/1 columns
        adds no value and diverges from the notebook's behaviour.
        """
        # Determine which columns to scale (all except binary/categorical flags)
        self._scale_cols = [
            c for c in X_train.columns
            if c not in self.config.skip_scale_cols
        ]
        self._scaler = StandardScaler()
        self._scaler.fit(X_train[self._scale_cols])
        return self

    # ── _apply_scaler ─────────────────────────────────────────────────────────

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to continuous columns only."""
        df = df.copy()
        df[self._scale_cols] = self._scaler.transform(df[self._scale_cols])
        return df

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """
        Apply the full feature-engineering pipeline to df.
        Lookup-based features are resolved from the fitted stores — NOT
        re-aggregated from df itself.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

        df = df.copy()

        df = self._add_temporal_features(df)
        df = self._add_amount_features(df)
        df = self._add_balance_features(df)
        df = self._add_geographic_features(df)
        df = self._add_velocity_features(df)
        df = self._add_behavioural_features(df)
        df = self._add_device_features(df)
        df = self._encode_categoricals(df)

        df.drop(
            columns=[c for c in self.config.drop_cols if c in df.columns],
            inplace=True,
        )

        if self.config.target_col in df.columns:
            df.drop(columns=[self.config.target_col], inplace=True)

        # Use selective scaler — binary flags are left unscaled
        if scale and self._scaler is not None:
            df = self._apply_scaler(df)

        return df

    # ── update_store ──────────────────────────────────────────────────────────

    def update_store(self, recent_transactions: pd.DataFrame) -> None:
        """
        Refresh sender and device lookup stores using a recent transaction
        window (e.g. last 90 days from production DB).
        """
        df = recent_transactions.copy()

        sender_agg = df.groupby("sender_id").agg(
            total_tx       =("transaction_id", "count"),
            unique_recv    =("receiver_id",    "nunique"),
            unique_devices =("device_id",      "nunique"),
            avg_amount     =("amount",         "mean"),
        )
        primary_device = (
            df.groupby(["sender_id", "device_id"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .drop_duplicates("sender_id")
            .set_index("sender_id")["device_id"]
            .rename("primary_device")
        )
        sender_agg = sender_agg.join(primary_device)
        self._sender_store = sender_agg.to_dict(orient="index")

        device_agg = df.groupby("device_id").agg(
            unique_senders=("sender_id", "nunique"),
        )
        self._device_store = device_agg.to_dict(orient="index")

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the entire fitted pipeline to a single .joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "FraudTransformationPipeline":
        """Deserialize a previously saved pipeline."""
        return joblib.load(path)

    # ── private feature-engineering helpers (unchanged from notebook) ─────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"]   = pd.to_datetime(df["timestamp"])
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"]       = df["timestamp"].dt.month
        df["is_night"]    = df["hour"].between(0, 5).astype(int)
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_amount"]    = np.log1p(df["amount"])
        df["is_high_value"] = (df["amount"] > 10_000).astype(int)
        sender_avg = df["sender_id"].map(
            lambda sid: self._sender_store.get(sid, {}).get(
                "avg_amount", self._fallbacks["avg_amount"]
            )
        )
        df["amount_vs_sender_avg"] = df["amount"] / (sender_avg + 1)
        return df

    def _add_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["balance_drain_rate"] = (
            df["amount"] / (df["sender_balance_before"] + 1)
        ).clip(0, 1)
        df["sender_balance_change"]   = df["sender_balance_before"] - df["sender_balance_after"]
        df["receiver_balance_change"] = df["receiver_balance_after"] - df["receiver_balance_before"]
        df["balance_discrepancy"]     = (
            (df["sender_balance_change"] - df["amount"]).abs() > 1
        ).astype(int)
        return df

    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lat1 = np.radians(self._NAIROBI_LAT)
        lat2 = np.radians(df["location_lat"].values)
        dlat = np.radians(df["location_lat"].values - self._NAIROBI_LAT)
        dlon = np.radians(df["location_lon"].values - self._NAIROBI_LON)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        df["dist_from_nairobi"] = 6371 * 2 * np.arcsin(np.sqrt(a))
        df["is_outside_kenya"]  = (
            (df["location_lat"] < -5) | (df["location_lat"] > 5) |
            (df["location_lon"] < 34) | (df["location_lon"] > 42)
        ).astype(int)
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sender_total_tx"] = df["sender_id"].map(
            lambda sid: self._sender_store.get(sid, {}).get(
                "total_tx", self._fallbacks["total_tx"]
            )
        )
        df["sender_unique_recv"] = df["sender_id"].map(
            lambda sid: self._sender_store.get(sid, {}).get(
                "unique_recv", self._fallbacks["unique_recv"]
            )
        )
        return df

    def _add_behavioural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sender_unique_devices"] = df["sender_id"].map(
            lambda sid: self._sender_store.get(sid, {}).get(
                "unique_devices", self._fallbacks["unique_devices"]
            )
        )

        def _device_switch(row):
            store_entry    = self._sender_store.get(row["sender_id"], {})
            primary_device = store_entry.get("primary_device", None)
            if primary_device is None:
                return self._fallbacks["is_device_switch"]
            return int(row["device_id"] != primary_device)

        df["is_device_switch"] = df.apply(_device_switch, axis=1)
        return df

    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["device_unique_senders"] = df["device_id"].map(
            lambda did: self._device_store.get(did, {}).get(
                "unique_senders", self._fallbacks["unique_senders"]
            )
        )
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["transaction_type_enc"] = (
            df["transaction_type"]
            .map(self.config.type_mapping)
            .fillna(-1)
            .astype(int)
        )
        return df


# ──────────────────────────────────────────────────────────────────────────────
# DataTransformation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class DataTransformation:
    """
    Orchestrates the full training workflow:
      1. Load validated data from parquet
      2. Fit the FraudTransformationPipeline on the full dataset
      3. Split into train / test (stratified)
      4. Transform both splits (feature engineering, no scaling yet)
      5. Scale FIRST — fit on train, apply to both splits
      6. Apply SMOTE to scaled training set only
      7. Save all artefacts to root_dir
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.pipeline = FraudTransformationPipeline(config)

    def run(self) -> None:
        try:
            root = self.config.root_dir
            root.mkdir(parents=True, exist_ok=True)

            # ── Step 1: Load ──────────────────────────────────────────────────
            if not self.config.data_path.exists():
                raise FileNotFoundError(
                    f"Validated data not found: {self.config.data_path}"
                )
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Loaded validated data: {df.shape}")

            # ── Step 2: Fit pipeline on full dataset ──────────────────────────
            self.pipeline.fit(df)
            logger.info("Pipeline fitted — sender and device stores built.")

            # ── Step 3: Stratified train/test split ───────────────────────────
            X = df.drop(columns=[self.config.target_col])
            y = df[self.config.target_col]

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                test_size    =self.config.test_size,
                stratify     =y,
                random_state =self.config.random_state,
            )
            logger.info(
                f"Train/test split — train: {X_train_raw.shape}, "
                f"test: {X_test_raw.shape}"
            )

            # ── Step 4: Feature engineering — no scaling yet ──────────────────
            X_train = self.pipeline.transform(X_train_raw, scale=False)
            X_test  = self.pipeline.transform(X_test_raw,  scale=False)
            logger.info("Feature engineering applied to train and test splits.")

            # ── Step 5: Scale BEFORE SMOTE (mirrors notebook order) ───────────
            # Scaler is fitted on real training data only, then applied to both.
            # SMOTE then interpolates in the already-scaled feature space —
            # synthetic samples are consistent with the scaler's distribution.
            self.pipeline.fit_scaler(X_train)
            X_train_scaled = self.pipeline._apply_scaler(X_train)
            X_test_scaled  = self.pipeline._apply_scaler(X_test)
            logger.info(
                f"StandardScaler fitted on {len(self.pipeline._scale_cols)} "
                f"continuous features and applied to both splits. "
                f"Unscaled (binary flags): {self.config.skip_scale_cols}"
            )

            # ── Step 6: SMOTE on scaled training set only ─────────────────────
            if self.config.smote:
                sm = SMOTE(
                    random_state=self.config.random_state,
                    k_neighbors=5,
                )
                X_train_np, y_train_np = sm.fit_resample(X_train_scaled, y_train)
                X_train_scaled = pd.DataFrame(
                    X_train_np, columns=X_train_scaled.columns
                )
                y_train = pd.Series(y_train_np, name=self.config.target_col)
                logger.info(
                    f"SMOTE applied — train shape: {X_train_scaled.shape}, "
                    f"fraud rate: {y_train.mean():.2%}"
                )

            # ── Step 7: Save artefacts ────────────────────────────────────────
            pipeline_path = root / "pipeline.joblib"
            self.pipeline.save(str(pipeline_path))
            logger.info(f"Transformation pipeline saved to {pipeline_path}")

            X_train_scaled.to_parquet(root / "X_train.parquet", index=False)
            X_test_scaled.to_parquet(root  / "X_test.parquet",  index=False)
            logger.info("X_train and X_test saved as parquet.")

            y_train.reset_index(drop=True).to_frame().to_parquet(
                root / "y_train.parquet", index=False
            )
            y_test.reset_index(drop=True).to_frame().to_parquet(
                root / "y_test.parquet", index=False
            )
            logger.info("y_train and y_test saved as parquet.")

            logger.info(
                f"Feature engineering complete — "
                f"features: {X_train_scaled.shape[1]}, "
                f"train samples: {X_train_scaled.shape[0]}, "
                f"test samples: {X_test_scaled.shape[0]}"
            )

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.DATA_TRANSFORMATION,
                severity=ErrorSeverity.HIGH
            )