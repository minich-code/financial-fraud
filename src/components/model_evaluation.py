# src/components/model_evaluation.py
import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config_manager.model_evaluation import ModelEvaluationConfig
from src.utils.exception import PipelineException, ErrorCategory, ErrorSeverity
from src.utils.logger import logger

# Threshold labels used consistently across plots and reports
THRESHOLD_LABELS = {
    0.3:  "T=0.30 (High Recall — Fraud Radar)",
    0.55: "T=0.55 (High Precision — Auto-Flag)",
}
THRESHOLD_COLORS = {}   # populated from config in __init__


class ModelEvaluator:
    """
    Evaluates the trained LightGBM model on the held-out test set.

    Produces:
      - evaluation_report.json     — full metrics at both thresholds
      - confusion_matrix_0.3.png   — confusion matrix at threshold 0.3
      - confusion_matrix_0.55.png  — confusion matrix at threshold 0.55
      - roc_curve.png              — ROC curve with both thresholds marked
      - precision_recall_curve.png — PR curve with both thresholds marked
      - classification_report_0.3.png
      - classification_report_0.55.png

    All artifacts are saved locally and logged to the existing MLflow
    training run (identified via run_id.json).
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.threshold_colors = {
            config.thresholds[0]: config.color_secondary,   # 0.3  → red
            config.thresholds[1]: config.color_tertiary,    # 0.55 → green
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate(self) -> None:
        try:
            model, X_test, y_test = self._load_artifacts()
            y_prob = model.predict_proba(X_test)[:, 1]

            report        = self._build_report(y_test, y_prob)
            plot_paths    = self._generate_plots(y_test, y_prob)

            self._save_report(report)
            self._log_to_mlflow(report, plot_paths)

            logger.info(
                f"Evaluation complete — "
                f"ROC-AUC: {report['threshold_independent']['roc_auc']} | "
                f"PR-AUC: {report['threshold_independent']['pr_auc']}"
            )

        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException(
                error=e,
                category=ErrorCategory.MODEL_VALIDATION,
                severity=ErrorSeverity.HIGH
            )

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self):
        for path in [
            self.config.model_path,
            self.config.X_test_path,
            self.config.y_test_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        model  = joblib.load(self.config.model_path)
        X_test = pd.read_parquet(self.config.X_test_path)
        y_test = pd.read_parquet(self.config.y_test_path).iloc[:, 0]

        logger.info(f"Test set loaded — {X_test.shape[0]} samples")
        return model, X_test, y_test

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _build_report(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Build the full evaluation report dict for both thresholds."""
        report: Dict[str, Any] = {
            "threshold_independent": {
                "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "pr_auc":  round(float(average_precision_score(y_test, y_prob)), 4),
            },
            "by_threshold": {},
        }

        for t in self.config.thresholds:
            y_pred = (y_prob >= t).astype(int)
            report["by_threshold"][str(t)] = {
                "threshold":         t,
                "label":             THRESHOLD_LABELS.get(t, f"T={t}"),
                "accuracy":          round(float(accuracy_score(y_test, y_pred)),   4),
                "precision":         round(float(precision_score(y_test, y_pred)),  4),
                "recall":            round(float(recall_score(y_test, y_pred)),     4),
                "f1":                round(float(f1_score(y_test, y_pred)),         4),
                "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred,
                    target_names=["Legitimate", "Fraud"],
                    output_dict=True,
                ),
            }
            logger.info(
                f"Threshold {t} — "
                f"Precision: {report['by_threshold'][str(t)]['precision']} | "
                f"Recall: {report['by_threshold'][str(t)]['recall']} | "
                f"F1: {report['by_threshold'][str(t)]['f1']}"
            )

        return report

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _generate_plots(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
    ) -> List[Path]:
        """Generate all plots and return their saved paths."""
        plt.style.use(self.config.plot_style)
        paths = []

        for t in self.config.thresholds:
            paths.append(self._plot_confusion_matrix(y_test, y_prob, t))
            paths.append(self._plot_classification_report(y_test, y_prob, t))

        paths.append(self._plot_roc_curve(y_test, y_prob))
        paths.append(self._plot_precision_recall_curve(y_test, y_prob))

        return paths

    def _plot_confusion_matrix(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
        threshold: float,
    ) -> Path:
        y_pred = (y_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_test, y_pred)

        # Normalised values for annotation (percentages)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 5.5))

        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax, cbar=False,
        )

        # Annotate each cell with count + percentage
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.5 else "#1e293b",
                )

        t_label = THRESHOLD_LABELS.get(threshold, f"T={threshold}")
        ax.set_title(
            f"Confusion Matrix — {t_label}",
            fontsize=14, fontweight="bold", pad=16
        )
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
        ax.set_ylabel("True Label", fontsize=12, labelpad=10)
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        path = self.config.root_dir / f"confusion_matrix_{threshold}.png"
        fig.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot → {path}")
        return path

    def _plot_roc_curve(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
    ) -> Path:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Main ROC curve
        ax.plot(
            fpr, tpr,
            color=self.config.color_primary,
            lw=2.5, label=f"LightGBM (AUC = {auc:.4f})"
        )

        # Random baseline
        ax.plot(
            [0, 1], [0, 1],
            color=self.config.color_diagonal,
            lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.50)"
        )

        # Mark each threshold point on the curve
        threshold_colors = [self.config.color_secondary, self.config.color_tertiary]
        for t, color in zip(self.config.thresholds, threshold_colors):
            idx = np.argmin(np.abs(roc_thresholds - t))
            ax.scatter(
                fpr[idx], tpr[idx],
                s=120, zorder=5, color=color,
                label=f"{THRESHOLD_LABELS.get(t, f'T={t}')} "
                      f"(FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f})",
            )
            ax.annotate(
                f"T={t}",
                xy=(fpr[idx], tpr[idx]),
                xytext=(fpr[idx] + 0.04, tpr[idx] - 0.04),
                fontsize=10, color=color, fontweight="bold",
            )

        ax.set_title("ROC Curve — LightGBM Fraud Detection", fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        path = self.config.root_dir / "roc_curve.png"
        fig.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ROC curve → {path}")
        return path

    def _plot_precision_recall_curve(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
    ) -> Path:
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        baseline = float(y_test.mean())

        fig, ax = plt.subplots(figsize=(8, 6))

        # Main PR curve
        ax.plot(
            recall_vals, precision_vals,
            color=self.config.color_primary,
            lw=2.5, label=f"LightGBM (PR-AUC = {pr_auc:.4f})"
        )

        # Random baseline (fraud rate)
        ax.axhline(
            baseline,
            color=self.config.color_diagonal,
            lw=1.5, linestyle="--",
            label=f"Random Baseline (Fraud Rate = {baseline:.2%})"
        )

        # Mark each threshold point
        threshold_colors = [self.config.color_secondary, self.config.color_tertiary]
        for t, color in zip(self.config.thresholds, threshold_colors):
            idx = np.argmin(np.abs(pr_thresholds - t))
            ax.scatter(
                recall_vals[idx], precision_vals[idx],
                s=120, zorder=5, color=color,
                label=f"{THRESHOLD_LABELS.get(t, f'T={t}')} "
                      f"(P={precision_vals[idx]:.3f}, R={recall_vals[idx]:.3f})",
            )
            ax.annotate(
                f"T={t}",
                xy=(recall_vals[idx], precision_vals[idx]),
                xytext=(recall_vals[idx] - 0.08, precision_vals[idx] + 0.03),
                fontsize=10, color=color, fontweight="bold",
            )

        ax.set_title("Precision-Recall Curve — LightGBM Fraud Detection", fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("Recall", fontsize=12, labelpad=10)
        ax.set_ylabel("Precision", fontsize=12, labelpad=10)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        path = self.config.root_dir / "precision_recall_curve.png"
        fig.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PR curve → {path}")
        return path

    def _plot_classification_report(
        self,
        y_test: pd.Series,
        y_prob: np.ndarray,
        threshold: float,
    ) -> Path:
        y_pred  = (y_prob >= threshold).astype(int)
        report  = classification_report(
            y_test, y_pred,
            target_names=["Legitimate", "Fraud"],
            output_dict=True
        )

        # Build a clean DataFrame for the heatmap
        rows     = ["Legitimate", "Fraud", "Macro Avg", "Weighted Avg"]
        row_keys = ["Legitimate", "Fraud", "macro avg", "weighted avg"]
        cols     = ["precision", "recall", "f1-score", "support"]

        data = []
        for key in row_keys:
            row_data = report.get(key, {})
            data.append([
                round(row_data.get("precision", 0), 3),
                round(row_data.get("recall",    0), 3),
                round(row_data.get("f1-score",  0), 3),
                int(row_data.get("support",     0)),
            ])

        df_report = pd.DataFrame(data, index=rows, columns=cols)

        # Separate support column (not a 0-1 metric)
        df_metrics = df_report[["precision", "recall", "f1-score"]].astype(float)
        df_support = df_report[["support"]]

        fig, axes = plt.subplots(
            1, 2, figsize=(10, 3.5),
            gridspec_kw={"width_ratios": [3, 1]}
        )

        # Metrics heatmap
        sns.heatmap(
            df_metrics, annot=True, fmt=".3f",
            cmap="Blues", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            ax=axes[0], cbar=False,
        )
        axes[0].set_title("Metrics", fontsize=12, fontweight="bold")
        axes[0].tick_params(axis="x", labelsize=10)
        axes[0].tick_params(axis="y", labelsize=10, rotation=0)

        # Support column
        sns.heatmap(
            df_support, annot=True, fmt=",d",
            cmap="Greys", linewidths=0.5, linecolor="white",
            ax=axes[1], cbar=False,
        )
        axes[1].set_title("Support", fontsize=12, fontweight="bold")
        axes[1].tick_params(axis="x", labelsize=10)
        axes[1].set_yticklabels([])

        t_label = THRESHOLD_LABELS.get(threshold, f"T={threshold}")
        fig.suptitle(
            f"Classification Report — {t_label}",
            fontsize=13, fontweight="bold", y=1.02
        )
        fig.tight_layout()

        path = self.config.root_dir / f"classification_report_{threshold}.png"
        fig.savefig(path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved classification report plot → {path}")
        return path

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_report(self, report: Dict[str, Any]) -> None:
        path = self.config.root_dir / "evaluation_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=4, default=str)
        logger.info(f"Evaluation report saved to {path}")

    def _log_to_mlflow(
        self,
        report: Dict[str, Any],
        plot_paths: List[Path],
    ) -> None:
        """Resume the training MLflow run and log all evaluation artifacts."""
        if not self.config.run_id_path.exists():
            raise FileNotFoundError(
                f"run_id.json not found at {self.config.run_id_path}. "
                "Ensure model training has completed successfully."
            )

        with open(self.config.run_id_path) as f:
            run_id = json.load(f)["run_id"]

        mlflow.set_tracking_uri(self.config.mlflow_uri)

        with mlflow.start_run(run_id=run_id):
            # Threshold-independent metrics
            ti = report["threshold_independent"]
            mlflow.log_metrics({
                "eval_roc_auc": ti["roc_auc"],
                "eval_pr_auc":  ti["pr_auc"],
            })

            # Per-threshold metrics with prefixed keys
            for t_str, t_metrics in report["by_threshold"].items():
                t_key = t_str.replace(".", "_")
                mlflow.log_metrics({
                    f"eval_t{t_key}_accuracy":  t_metrics["accuracy"],
                    f"eval_t{t_key}_precision": t_metrics["precision"],
                    f"eval_t{t_key}_recall":    t_metrics["recall"],
                    f"eval_t{t_key}_f1":        t_metrics["f1"],
                })

            # Log all plots as MLflow artifacts
            for path in plot_paths:
                mlflow.log_artifact(str(path), artifact_path="evaluation_plots")

            # Log the full report JSON
            mlflow.log_artifact(
                str(self.config.root_dir / "evaluation_report.json"),
                artifact_path="evaluation_report"
            )

        logger.info(f"Evaluation artifacts logged to MLflow run {run_id}")