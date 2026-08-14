"""Metrics, threshold selection and reporting."""

from __future__ import annotations

from term_deposit.evaluation.metrics import (
    ClassificationMetrics,
    binary_metrics,
    bootstrap_interval,
    calibration_summary,
    lift_at_k,
    precision_at_k,
    recall_at_k,
    within_period_metrics,
)
from term_deposit.evaluation.report import (
    EvaluationReport,
    ScorableModel,
    build_comparison_table,
    evaluate_all,
    evaluate_artifact,
    evaluate_model,
    select_best,
    write_reports,
)
from term_deposit.evaluation.thresholds import (
    ThresholdChoice,
    expected_value,
    select_threshold,
    threshold_sweep,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationReport",
    "ScorableModel",
    "ThresholdChoice",
    "binary_metrics",
    "bootstrap_interval",
    "build_comparison_table",
    "calibration_summary",
    "evaluate_all",
    "evaluate_artifact",
    "evaluate_model",
    "expected_value",
    "lift_at_k",
    "precision_at_k",
    "recall_at_k",
    "select_best",
    "select_threshold",
    "threshold_sweep",
    "within_period_metrics",
    "write_reports",
]
