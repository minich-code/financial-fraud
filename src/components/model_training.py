# src/components/model_training.py
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config_manager.model_training import ModelTrainingConfig
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

# Suppress Optuna per-trial noise — we log summaries ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Trains a LightGBM fraud detection model with two modes:

    Default mode  (run_hyperparameter_search: false)
        Loads params from best_params.json if it exists, otherwise
        falls back to default_params from config. Trains once and logs
        everything to MLflow.

    Optuna mode   (run_hyperparameter_search: true)
        Runs a hyperparameter search, saves best_params.json, then
        trains the final model with those params. All trials and the
        final run are logged to MLflow.
    """

    BEST_PARAMS_FILENAME = "best_params.json"
    MODEL_FILENAME = "lgb_model.joblib"

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self._best_params_path = config.root_dir / self.BEST_PARAMS_FILENAME

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> None:
        try:
            X_train, X_test, y_train, y_test = self._load_data()

            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.experiment_name)

            if self.config.run_hyperparameter_search:
                logger.info("Hyperparameter search enabled — running Optuna.")
                params = self._run_optuna(X_train, y_train)
            else:
                params = self._load_params()
                logger.info(
                    f"Using {'saved best' if self._best_params_path.exists() else 'default'} "
                    f"params: {params}"
                )

            self._train_and_log(X_train, X_test, y_train, y_test, params)

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.MODEL_TRAINING,
                severity=ErrorSeverity.HIGH
            )

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load feature matrices and targets from parquet."""
        for path in [
            self.config.X_train_path, self.config.X_test_path,
            self.config.y_train_path, self.config.y_test_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        X_train = pd.read_parquet(self.config.X_train_path)
        X_test  = pd.read_parquet(self.config.X_test_path)
        y_train = pd.read_parquet(self.config.y_train_path)[self.config.target_col]
        y_test  = pd.read_parquet(self.config.y_test_path)[self.config.target_col]

        logger.info(
            f"Data loaded — train: {X_train.shape}, test: {X_test.shape}, "
            f"fraud rate (train): {y_train.mean():.2%}"
        )
        return X_train, X_test, y_train, y_test

    # ── Param resolution ──────────────────────────────────────────────────────

    def _load_params(self) -> Dict[str, Any]:
        """
        Return best_params.json if it exists (saved by a previous Optuna run),
        otherwise fall back to default_params from config.
        """
        if self._best_params_path.exists():
            with open(self._best_params_path) as f:
                return json.load(f)
        return self.config.default_params

    def _save_best_params(self, params: Dict[str, Any]) -> None:
        with open(self._best_params_path, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Best params saved to {self._best_params_path}")

    # ── Optuna hyperparameter search ──────────────────────────────────────────

    def _run_optuna(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Run Optuna study. Each trial is logged as a nested MLflow run.
        Returns the best params dict (fixed + tuned), saves to best_params.json.
        """
        optuna_cfg    = self.config.optuna
        fixed_params  = dict(optuna_cfg["fixed_params"])
        search_space  = dict(optuna_cfg["search_space"])
        cv_folds      = int(optuna_cfg["cv_folds"])
        scoring       = str(optuna_cfg["scoring"])

        with mlflow.start_run(run_name="optuna_search", nested=False) as parent_run:
            mlflow.set_tag("mode", "optuna_search")

            def objective(trial: optuna.Trial) -> float:
                tuned = {
                    "num_leaves":        trial.suggest_int(
                        "num_leaves",
                        search_space["num_leaves"]["low"],
                        search_space["num_leaves"]["high"]
                    ),
                    "learning_rate":     trial.suggest_float(
                        "learning_rate",
                        search_space["learning_rate"]["low"],
                        search_space["learning_rate"]["high"],
                        log=search_space["learning_rate"].get("log", False)
                    ),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples",
                        search_space["min_child_samples"]["low"],
                        search_space["min_child_samples"]["high"]
                    ),
                    "subsample":         trial.suggest_float(
                        "subsample",
                        search_space["subsample"]["low"],
                        search_space["subsample"]["high"]
                    ),
                    "colsample_bytree":  trial.suggest_float(
                        "colsample_bytree",
                        search_space["colsample_bytree"]["low"],
                        search_space["colsample_bytree"]["high"]
                    ),
                    "scale_pos_weight":  trial.suggest_float(
                        "scale_pos_weight",
                        search_space["scale_pos_weight"]["low"],
                        search_space["scale_pos_weight"]["high"]
                    ),
                }
                params = {**fixed_params, **tuned}
                model  = lgb.LGBMClassifier(**params)
                cv     = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True,
                    random_state=fixed_params.get("random_state", 42)
                )
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=-1
                )

                # Log each trial as a nested run
                with mlflow.start_run(
                    run_name=f"trial_{trial.number}",
                    nested=True,
                    tags={"parent_run_id": parent_run.info.run_id}
                ):
                    mlflow.log_params(tuned)
                    mlflow.log_metric(f"cv_{scoring}", scores.mean())

                return scores.mean()

            study = optuna.create_study(
                direction=str(optuna_cfg["direction"])
            )
            study.optimize(
                objective,
                n_trials=int(optuna_cfg["n_trials"]),
                show_progress_bar=True
            )

            logger.info(
                f"Optuna complete — best CV {scoring}: "
                f"{study.best_value:.4f} | params: {study.best_params}"
            )
            mlflow.log_metric(f"best_cv_{scoring}", study.best_value)
            mlflow.log_params(study.best_params)

        # Merge fixed + tuned into final params and persist
        best_params = {**fixed_params, **study.best_params}
        self._save_best_params(best_params)
        return best_params

    # ── Training and MLflow logging ───────────────────────────────────────────

    def _train_and_log(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
        params:  Dict[str, Any],
    ) -> None:
        """Train LightGBM with given params, evaluate, log to MLflow, save model."""
        run_name = (
            "lgb_optuna_final"
            if self.config.run_hyperparameter_search
            else "lgb_default"
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag(
                "mode",
                "optuna_final" if self.config.run_hyperparameter_search else "default"
            )

            # Train
            t0    = time.time()
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            logger.info(f"Model trained in {elapsed:.1f}s")

            # Evaluate
            metrics = self._evaluate(model, X_test, y_test)
            logger.info(
                f"Evaluation — AUC-ROC: {metrics['roc_auc']:.4f} | "
                f"PR-AUC: {metrics['pr_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("training_time_s", round(elapsed, 2))
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(model, artifact_path="lgb_model")

            # Save model locally
            model_path = self.config.root_dir / self.MODEL_FILENAME
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

            run_id_path = self.config.root_dir / "run_id.json"
            with open(run_id_path, "w") as f:
                json.dump({"run_id": mlflow.active_run().info.run_id}, f, indent=4)
            logger.info(f"MLflow run ID saved to {run_id_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(
        self, model: lgb.LGBMClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Compute all evaluation metrics on the test set."""
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        return {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)),        4),
            "f1":        round(float(f1_score(y_test, y_pred)),               4),
            "precision": round(float(precision_score(y_test, y_pred)),        4),
            "recall":    round(float(recall_score(y_test, y_pred)),           4),
            "roc_auc":   round(float(roc_auc_score(y_test, y_prob)),          4),
            "pr_auc":    round(float(average_precision_score(y_test, y_prob)),4),
        }