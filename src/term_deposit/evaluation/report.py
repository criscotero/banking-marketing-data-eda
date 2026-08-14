"""Turn trained models into the reports a reviewer can act on.

An evaluation is not one number. Each model produces:

* threshold-free ranking metrics (average precision, ROC-AUC),
* capacity metrics (precision/lift at k) at the fractions a campaign can call,
* calibration metrics, because expected-value decisions need real probabilities,
* the operating point chosen on validation and its confusion matrix on test,
* within-period metrics, which strip out the calendar signal (ADR 0003),
* and cross-validation and backtest summaries where they were computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from term_deposit.config import AppConfig
from term_deposit.data.splits import DataSplit
from term_deposit.evaluation.metrics import (
    ClassificationMetrics,
    binary_metrics,
    bootstrap_interval,
    calibration_summary,
    comparison_frame,
    metrics_to_row,
    within_period_metrics,
)
from term_deposit.evaluation.thresholds import ThresholdChoice, select_threshold, threshold_sweep
from term_deposit.utils.logging import get_logger
from term_deposit.utils.serialization import write_json

logger = get_logger(__name__)


@runtime_checkable
class ScorableModel(Protocol):
    """What evaluation needs from a trained model.

    Declared structurally rather than importing
    :class:`~term_deposit.training.trainer.TrainedModel`, so that ``evaluation``
    stays a leaf of the dependency graph: training imports evaluation, never the
    other way round. Anything with these four attributes can be evaluated,
    including a stub in a test.
    """

    @property
    def name(self) -> str:
        """Identifier used in reports and filenames.

        Declared read-only so that a frozen dataclass such as
        :class:`~term_deposit.training.trainer.TrainedModel` satisfies it.
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Positive-class probabilities."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Everything measured about one model under one protocol."""

    model_name: str
    split_strategy: str
    threshold: ThresholdChoice
    test_metrics: ClassificationMetrics
    validation_metrics: ClassificationMetrics | None
    within_period: dict[str, Any]
    bootstrap: dict[str, float]
    calibration: pd.DataFrame
    sweep: pd.DataFrame
    cross_validation: dict[str, Any] | None = None
    backtest: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, used for ``reports/metrics/*.json``."""
        return {
            "model_name": self.model_name,
            "split_strategy": self.split_strategy,
            "threshold": self.threshold.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "validation_metrics": (
                self.validation_metrics.to_dict() if self.validation_metrics else None
            ),
            "within_period": self.within_period,
            "bootstrap_average_precision": self.bootstrap,
            "calibration_bins": self.calibration.to_dict(orient="records"),
            "cross_validation": self.cross_validation,
            "backtest": self.backtest,
            "extras": self.extras,
        }

    def headline(self) -> str:
        """One-line summary for logs and the console."""
        within = self.within_period.get("weighted_roc_auc", float("nan"))
        return (
            f"{self.model_name}: {self.test_metrics.headline()} within-period ROC-AUC={within:.4f}"
        )


def evaluate_model(
    model: ScorableModel,
    split: DataSplit,
    config: AppConfig,
) -> EvaluationReport:
    """Evaluate one trained model on the test split.

    The threshold is chosen on the validation split and then applied to test,
    unchanged. When there is no usable validation split — which happens with a
    zero-size validation fraction — the threshold falls back to the capacity rule
    so that it is still never fitted on test labels.

    Args:
        model: A fitted model.
        split: The split it was trained on.
        config: Resolved application configuration.

    Returns:
        An :class:`EvaluationReport`.
    """
    evaluation = config.evaluation
    has_validation = len(split.y_validation) > 0 and split.y_validation.nunique() >= 2

    if has_validation:
        validation_scores = model.predict_proba(split.X_validation)
        threshold = select_threshold(
            split.y_validation,
            validation_scores,
            objective=evaluation.threshold_objective,
            value_per_subscription=evaluation.value_per_subscription,
            cost_per_call=evaluation.cost_per_call,
            capacity_fraction=evaluation.capacity_fraction,
            chosen_on="validation",
        )
        validation_metrics: ClassificationMetrics | None = binary_metrics(
            split.y_validation,
            validation_scores,
            threshold=threshold.threshold,
            top_k_fractions=evaluation.top_k_fractions,
        )
    else:
        logger.warning(
            "No usable validation split for %s; falling back to a capacity-based "
            "threshold on the training split",
            model.name,
        )
        training_scores = model.predict_proba(split.X_train)
        threshold = select_threshold(
            split.y_train,
            training_scores,
            objective="top_k",
            capacity_fraction=evaluation.capacity_fraction,
            chosen_on="train",
        )
        validation_metrics = None

    test_scores = model.predict_proba(split.X_test)
    test_metrics = binary_metrics(
        split.y_test,
        test_scores,
        threshold=threshold.threshold,
        top_k_fractions=evaluation.top_k_fractions,
    )

    within: dict[str, Any] = {}
    if evaluation.report_within_period:
        within = within_period_metrics(
            split.y_test,
            test_scores,
            split.periods_for("test").astype(str),
            min_rows=evaluation.within_period_min_rows,
        )

    cross_validation = getattr(model, "cross_validation", None)
    backtest = getattr(model, "backtest", None)

    report = EvaluationReport(
        model_name=model.name,
        split_strategy=split.strategy,
        threshold=threshold,
        test_metrics=test_metrics,
        validation_metrics=validation_metrics,
        within_period=within,
        bootstrap=bootstrap_interval(
            split.y_test,
            test_scores,
            metric=evaluation.primary_metric,
            n_resamples=evaluation.n_bootstrap,
            seed=config.random_state,
        ),
        calibration=calibration_summary(split.y_test, test_scores),
        sweep=threshold_sweep(
            split.y_test,
            test_scores,
            value_per_subscription=evaluation.value_per_subscription,
            cost_per_call=evaluation.cost_per_call,
        ),
        cross_validation=(cross_validation.to_dict() if cross_validation is not None else None),
        backtest=(
            {"summary": backtest.summary(), "folds": list(backtest.folds)}
            if backtest is not None
            else None
        ),
        extras={
            "calibrated": getattr(model, "calibrator", None) is not None,
            "fit_seconds": getattr(model, "fit_seconds", float("nan")),
            **getattr(model, "extras", {}),
        },
    )
    logger.info(report.headline())
    return report


def evaluate_all(
    models: Sequence[ScorableModel],
    split: DataSplit,
    config: AppConfig,
) -> list[EvaluationReport]:
    """Evaluate every trained model on the same split."""
    return [evaluate_model(model, split, config) for model in models]


def build_comparison_table(
    reports: Sequence[EvaluationReport],
    *,
    primary_metric: str = "average_precision",
) -> pd.DataFrame:
    """Assemble a model-comparison table sorted by the primary metric.

    Includes the within-period column next to the pooled one so that the gap
    between them is visible in the table a reader looks at first.
    """
    rows = []
    for report in reports:
        extra: dict[str, Any] = {
            "split": report.split_strategy,
            "within_period_roc_auc": report.within_period.get("weighted_roc_auc", float("nan")),
            "within_period_ap": report.within_period.get(
                "weighted_average_precision", float("nan")
            ),
            "roc_auc_inflation": report.within_period.get("roc_auc_inflation", float("nan")),
            "calibrated": report.extras.get("calibrated", False),
        }
        if report.cross_validation:
            extra["cv_mean"] = report.cross_validation["mean"]
            extra["cv_std"] = report.cross_validation["std"]
        if report.backtest and report.backtest.get("summary"):
            summary = report.backtest["summary"]
            extra["backtest_ap_mean"] = summary.get("average_precision_mean", float("nan"))
            extra["backtest_roc_auc_mean"] = summary.get("roc_auc_mean", float("nan"))
            extra["backtest_lift20_mean"] = summary.get("lift_at_20pct_mean", float("nan"))
        rows.append(metrics_to_row(report.model_name, report.test_metrics, **extra))
    return comparison_frame(rows, sort_by=primary_metric)


def select_best(
    reports: Sequence[EvaluationReport],
    *,
    primary_metric: str = "average_precision",
    prefer_backtest: bool = True,
) -> EvaluationReport:
    """Pick the model to ship.

    Prefers the mean backtest score when available: a single held-out window is
    one draw from a noisy distribution, while the backtest averages over every
    trailing month and so is much less sensitive to which month happens to be last.

    Args:
        reports: Candidate reports.
        primary_metric: Metric name, used for the single-window fallback.
        prefer_backtest: Use the backtest mean when every candidate has one.

    Returns:
        The winning report.

    Raises:
        ValueError: If ``reports`` is empty.
    """
    if not reports:
        msg = "cannot select a best model from an empty report list"
        raise ValueError(msg)

    def backtest_score(report: EvaluationReport) -> float:
        if not report.backtest or not report.backtest.get("summary"):
            return float("nan")
        return float(report.backtest["summary"].get("average_precision_mean", float("nan")))

    if prefer_backtest:
        scores = [backtest_score(report) for report in reports]
        if all(np.isfinite(score) for score in scores):
            best = reports[int(np.argmax(scores))]
            logger.info(
                "Selected %s by mean backtest average precision (%.4f)",
                best.model_name,
                max(scores),
            )
            return best

    def single_window_score(report: EvaluationReport) -> float:
        value = getattr(report.test_metrics, primary_metric, float("nan"))
        return float(value) if np.isfinite(value) else -np.inf

    best = max(reports, key=single_window_score)
    logger.info(
        "Selected %s by test-set %s (%.4f)",
        best.model_name,
        primary_metric,
        single_window_score(best),
    )
    return best


def write_reports(
    reports: Sequence[EvaluationReport],
    output_dir: Path,
    *,
    primary_metric: str = "average_precision",
) -> dict[str, Path]:
    """Write per-model JSON, a comparison table and per-period breakdowns.

    Returns:
        A mapping of logical name to written path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for report in reports:
        slug = report.model_name.lower().replace(" ", "-")
        written[f"metrics::{report.model_name}"] = write_json(
            output_dir / f"{report.split_strategy}__{slug}.json", report.to_dict()
        )

    table = build_comparison_table(reports, primary_metric=primary_metric)
    strategy = reports[0].split_strategy if reports else "unknown"
    csv_path = output_dir / f"{strategy}__comparison.csv"
    table.to_csv(csv_path, index=False)
    written["comparison_csv"] = csv_path
    written["comparison_json"] = write_json(
        output_dir / f"{strategy}__comparison.json", table.to_dict(orient="records")
    )

    period_rows = [
        {"model": report.model_name, **entry}
        for report in reports
        for entry in report.within_period.get("by_period", [])
    ]
    if period_rows:
        period_path = output_dir / f"{strategy}__within_period.csv"
        pd.DataFrame(period_rows).to_csv(period_path, index=False)
        written["within_period_csv"] = period_path

    backtest_rows = [
        {"model": report.model_name, **fold}
        for report in reports
        if report.backtest
        for fold in report.backtest.get("folds", [])
    ]
    if backtest_rows:
        backtest_path = output_dir / f"{strategy}__backtest.csv"
        pd.DataFrame(backtest_rows).to_csv(backtest_path, index=False)
        written["backtest_csv"] = backtest_path

    logger.info("Wrote %d report file(s) to %s", len(written), output_dir)
    return written


def evaluate_artifact(
    artifact: Any,
    frame: pd.DataFrame,
    labels: pd.Series,
    config: AppConfig,
    *,
    periods: pd.Series | None = None,
) -> EvaluationReport:
    """Score a saved artifact against a labelled dataset.

    Used by ``scripts/evaluate.py`` to re-check a persisted model without
    retraining, which is the only way to confirm that the artifact and the
    reported metrics actually correspond.
    """
    scores = np.asarray(artifact.pipeline.predict_proba(frame))[:, 1]
    threshold = float(artifact.metadata.decision_threshold)
    metrics = binary_metrics(
        labels, scores, threshold=threshold, top_k_fractions=config.evaluation.top_k_fractions
    )
    within = (
        within_period_metrics(
            labels, scores, periods.astype(str), min_rows=config.evaluation.within_period_min_rows
        )
        if periods is not None
        else {}
    )
    return EvaluationReport(
        model_name=artifact.metadata.model_name,
        split_strategy=artifact.metadata.split_strategy,
        threshold=ThresholdChoice(
            threshold=threshold,
            objective=artifact.metadata.threshold_objective,
            objective_value=float("nan"),
            selected_fraction=float((scores >= threshold).mean()),
            expected_value_per_contact=float("nan"),
            chosen_on="loaded-from-artifact",
        ),
        test_metrics=metrics,
        validation_metrics=None,
        within_period=within,
        bootstrap=bootstrap_interval(
            labels,
            scores,
            metric=config.evaluation.primary_metric,
            n_resamples=config.evaluation.n_bootstrap,
            seed=config.random_state,
        ),
        calibration=calibration_summary(labels, scores),
        sweep=threshold_sweep(
            labels,
            scores,
            value_per_subscription=config.evaluation.value_per_subscription,
            cost_per_call=config.evaluation.cost_per_call,
        ),
        extras={"source": "artifact", "artifact_created_at": artifact.metadata.created_at},
    )
