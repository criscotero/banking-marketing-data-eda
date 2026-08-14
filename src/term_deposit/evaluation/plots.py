"""Figures for the reports and notebooks.

Matplotlib is imported lazily so that the training and inference paths stay
importable without the ``viz`` extra — a scoring container has no business
carrying a plotting stack.

Every function returns the ``Figure`` it drew, so a notebook can keep working on
it and a script can save and close it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from term_deposit.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

logger = get_logger(__name__)

_MISSING_MATPLOTLIB = (
    "matplotlib is required for plotting. Install the viz extra: `uv sync --extra viz`."
)


def _pyplot() -> Any:
    """Import pyplot with a non-interactive backend, or explain what is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(_MISSING_MATPLOTLIB) from error
    return plt


def save_figure(figure: Figure, path: Path, *, dpi: int = 150, close: bool = True) -> Path:
    """Write a figure to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        _pyplot().close(figure)
    logger.debug("Wrote figure %s", path)
    return path


def plot_roc_curves(scores: Mapping[str, np.ndarray], y_true: Any) -> Figure:
    """Overlay ROC curves for several models."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(6.5, 6))
    for name, values in scores.items():
        fpr, tpr, _ = roc_curve(y_true, values)
        axes.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, values):.3f})")
    axes.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axes.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(alpha=0.3)
    return cast("Figure", figure)


def plot_precision_recall_curves(scores: Mapping[str, np.ndarray], y_true: Any) -> Figure:
    """Overlay precision-recall curves.

    The legend reports **average precision** — the area under *this* curve. The
    original notebook labelled its PR curves with the previously computed
    ROC-AUC, which made the plot read as if it showed a much stronger result.
    """
    plt = _pyplot()
    labels = np.asarray(y_true).ravel()
    base_rate = float(labels.mean())

    figure, axes = plt.subplots(figsize=(6.5, 6))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        axes.plot(
            recall, precision, label=f"{name} (AP={average_precision_score(labels, values):.3f})"
        )
    axes.axhline(
        base_rate, color="k", linestyle="--", linewidth=1, label=f"Base rate ({base_rate:.3f})"
    )
    axes.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves",
        ylim=(0.0, 1.02),
    )
    axes.legend(loc="upper right", fontsize=9)
    axes.grid(alpha=0.3)
    return cast("Figure", figure)


def plot_confusion_matrix(y_true: Any, y_score: Any, threshold: float, title: str) -> Figure:
    """Confusion matrix at an explicit threshold, annotated with counts and shares."""
    plt = _pyplot()
    predictions = (np.asarray(y_score) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    figure, axes = plt.subplots(figsize=(4.6, 4.2))
    axes.imshow(matrix, cmap="Blues")
    total = matrix.sum()
    for row in range(2):
        for column in range(2):
            count = matrix[row, column]
            axes.text(
                column,
                row,
                f"{count:,}\n{count / total:.1%}",
                ha="center",
                va="center",
                color="white" if count > matrix.max() / 2 else "black",
                fontsize=11,
            )
    axes.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{title}\n(threshold = {threshold:.3f})",
    )
    return cast("Figure", figure)


def plot_calibration(scores: Mapping[str, np.ndarray], y_true: Any, *, n_bins: int = 10) -> Figure:
    """Reliability diagram: mean predicted probability against observed frequency.

    Points below the diagonal mean the model is over-confident, which is exactly
    what class weighting produces and why the pipeline calibrates on validation.
    """
    plt = _pyplot()
    labels = np.asarray(y_true).ravel()

    figure, axes = plt.subplots(figsize=(6, 6))
    for name, values in scores.items():
        frame = pd.DataFrame({"y": labels, "p": np.asarray(values).ravel()})
        frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop", labels=False)
        grouped = frame.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
        axes.plot(grouped["p"], grouped["y"], marker="o", label=name)
    axes.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes.set(
        xlabel="Mean predicted probability",
        ylabel="Observed subscription rate",
        title="Calibration (equal-count bins)",
    )
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3)
    return cast("Figure", figure)


def plot_lift_curve(scores: Mapping[str, np.ndarray], y_true: Any, *, n_points: int = 50) -> Figure:
    """Cumulative lift against the share of the list called.

    The operational chart: read off how much better than random the campaign does
    when it works the top *x*% of the ranking.
    """
    plt = _pyplot()
    labels = np.asarray(y_true).ravel()
    base_rate = float(labels.mean())
    fractions = np.linspace(0.02, 1.0, n_points)

    figure, axes = plt.subplots(figsize=(6.5, 5))
    for name, values in scores.items():
        order = np.argsort(-np.asarray(values).ravel(), kind="stable")
        ordered = labels[order]
        lifts = [ordered[: max(1, round(len(ordered) * f))].mean() / base_rate for f in fractions]
        axes.plot(fractions * 100, lifts, label=name)
    axes.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Random calling")
    axes.set(
        xlabel="Share of the list called (%)",
        ylabel="Lift over the base rate",
        title="Cumulative lift",
    )
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3)
    return cast("Figure", figure)


def plot_period_drift(summary: pd.DataFrame) -> Figure:
    """Subscription rate and macro indicators per calendar month.

    The single most important chart in the project: the base rate and the macro
    block move together, which is why a random split lets a model recover one
    from the other.
    """
    plt = _pyplot()
    periods = summary["contact_period"].astype(str)

    figure, axes = plt.subplots(figsize=(11, 5))
    axes.bar(
        periods,
        summary["subscription_rate"],
        color="#4C78A8",
        alpha=0.85,
        label="Subscription rate",
    )
    axes.set_ylabel("Subscription rate", color="#4C78A8")
    axes.tick_params(axis="x", rotation=90, labelsize=8)
    axes.set_title("Subscription rate and macro indicators by contact month")

    if "euribor3m" in summary.columns:
        twin = axes.twinx()
        twin.plot(
            periods,
            summary["euribor3m"],
            color="#E45756",
            marker="o",
            linewidth=2,
            label="euribor3m",
        )
        twin.set_ylabel("euribor3m (%)", color="#E45756")

    axes.grid(axis="y", alpha=0.3)
    return cast("Figure", figure)


def plot_within_vs_pooled(reports: Sequence[Any]) -> Figure:
    """Pooled versus within-period ROC-AUC, per model.

    The gap between the pair of bars is the share of the headline metric that
    comes from telling calendar months apart rather than customers.
    """
    plt = _pyplot()
    names = [report.model_name for report in reports]
    pooled = [report.test_metrics.roc_auc for report in reports]
    within = [report.within_period.get("weighted_roc_auc", np.nan) for report in reports]

    positions = np.arange(len(names))
    width = 0.38

    figure, axes = plt.subplots(figsize=(max(6.5, 1.6 * len(names)), 5))
    axes.bar(positions - width / 2, pooled, width, label="Pooled ROC-AUC", color="#4C78A8")
    axes.bar(positions + width / 2, within, width, label="Within-period ROC-AUC", color="#F58518")
    axes.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Random ranking")
    axes.set(
        xticks=positions,
        ylabel="ROC-AUC",
        title="How much of the score survives when the calendar is held fixed",
        ylim=(0.4, 1.0),
    )
    axes.set_xticklabels(names, rotation=20, ha="right")
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.3)
    return cast("Figure", figure)


def plot_backtest(folds: pd.DataFrame) -> Figure:
    """Per-fold backtest performance over time, one line per model."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(10, 5))
    for name, group in folds.groupby("model", observed=True):
        ordered = group.sort_values("period")
        axes.plot(
            ordered["period"].astype(str), ordered["average_precision"], marker="o", label=name
        )
    if "base_rate" in folds.columns:
        reference = folds.drop_duplicates("period").sort_values("period")
        axes.plot(
            reference["period"].astype(str),
            reference["base_rate"],
            "k--",
            linewidth=1.5,
            label="Base rate (no-skill AP)",
        )
    axes.set(
        xlabel="Test month",
        ylabel="Average precision",
        title="Rolling-origin backtest: train on all prior months, score the next",
    )
    axes.tick_params(axis="x", rotation=60, labelsize=8)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3)
    return cast("Figure", figure)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen_threshold: float) -> Figure:
    """Precision, recall and expected value across thresholds, with the choice marked."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.5, 5))
    axes.plot(sweep["threshold"], sweep["precision"], label="Precision")
    axes.plot(sweep["threshold"], sweep["recall"], label="Recall")
    axes.plot(sweep["threshold"], sweep["f1"], label="F1")
    axes.axvline(
        chosen_threshold,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label=f"Chosen ({chosen_threshold:.3f})",
    )
    axes.set(xlabel="Decision threshold", ylabel="Score", title="Threshold trade-off")
    axes.legend(loc="upper left", fontsize=9)
    axes.grid(alpha=0.3)

    twin = axes.twinx()
    twin.plot(
        sweep["threshold"],
        sweep["expected_value_per_contact"],
        color="#54A24B",
        linestyle=":",
        linewidth=2,
    )
    twin.set_ylabel("Expected value per contact", color="#54A24B")
    return cast("Figure", figure)


def plot_feature_importance(importances: pd.DataFrame, *, top_n: int = 20, title: str) -> Figure:
    """Horizontal bar chart of the top ``top_n`` features."""
    plt = _pyplot()
    top = importances.head(top_n).iloc[::-1]
    figure, axes = plt.subplots(figsize=(7.5, max(4, 0.32 * len(top))))
    axes.barh(top["feature"], top["importance"], color="#4C78A8")
    axes.set(xlabel="Importance", title=title)
    axes.grid(axis="x", alpha=0.3)
    return cast("Figure", figure)
