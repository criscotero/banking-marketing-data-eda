"""Metrics chosen for a ranked, capacity-constrained, imbalanced problem.

Three deliberate departures from the usual accuracy/ROC-AUC report:

* **Average precision, not ROC-AUC, is primary.** With an 11% base rate, ROC-AUC
  is dominated by how the model orders the majority class. Average precision
  tracks the positives, which is what the campaign is paying for.
* **Lift and precision at a capacity fraction are reported.** A call centre works
  a fixed-length list. "How many subscribers are in the top 20% of the ranking"
  is the operational question; F1 at an arbitrary 0.5 threshold is not.
* **Within-period metrics are reported alongside pooled ones.** Pooling across
  calendar months lets a model score points for knowing *when* a row is from.
  A live campaign ranks customers inside one month, so within-period scores are
  the deployment-relevant number. See ADR 0003.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from term_deposit.utils.logging import get_logger
from term_deposit.utils.seeding import make_rng

logger = get_logger(__name__)

DEFAULT_TOP_K = (0.05, 0.10, 0.20, 0.30)


def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned 1-D float/int arrays."""
    true_array = np.asarray(y_true).ravel()
    score_array = np.asarray(y_score, dtype=float).ravel()
    if true_array.shape != score_array.shape:
        msg = f"y_true and y_score must align: {true_array.shape} vs {score_array.shape}"
        raise ValueError(msg)
    if true_array.size == 0:
        msg = "cannot compute metrics on an empty array"
        raise ValueError(msg)
    return true_array.astype(int), score_array


#: Seed for the tie-break permutation. Fixed so that top-k metrics are
#: reproducible, and independent of the data so that it carries no signal.
_TIE_BREAK_SEED = 20_240_101


def _rank_order(y_score: np.ndarray) -> np.ndarray:
    """Descending score order with a deterministic random tie-break.

    Ranking by a stable sort would break ties on original row position — and in
    this dataset rows are ordered by contact date, so a model that emits many
    identical scores would silently be ranked by *time*. That inflates top-k
    metrics for exactly the models that deserve it least: the constant baseline
    scores a lift above 1.0 purely because later months have a higher base rate.

    A fixed-seed permutation removes that artefact while keeping the metric
    reproducible across runs.
    """
    tie_break = make_rng(_TIE_BREAK_SEED).permutation(y_score.size)
    # lexsort takes the last key as primary.
    return np.lexsort((tie_break, -y_score))


def top_k_count(n_rows: int, fraction: float) -> int:
    """Number of rows in the top ``fraction`` of a list of ``n_rows``."""
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must be in (0, 1], got {fraction}"
        raise ValueError(msg)
    return max(1, round(n_rows * fraction))


def precision_at_k(y_true: Any, y_score: Any, fraction: float) -> float:
    """Share of true positives among the highest-scored ``fraction`` of rows.

    Directly interpretable: the hit rate an agent would experience while working
    the top of the list.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    k = top_k_count(true_array.size, fraction)
    selected = _rank_order(score_array)[:k]
    return float(true_array[selected].mean())


def recall_at_k(y_true: Any, y_score: Any, fraction: float) -> float:
    """Share of all positives captured within the top ``fraction`` of rows."""
    true_array, score_array = _as_arrays(y_true, y_score)
    positives = int(true_array.sum())
    if positives == 0:
        return float("nan")
    k = top_k_count(true_array.size, fraction)
    selected = _rank_order(score_array)[:k]
    return float(true_array[selected].sum() / positives)


def lift_at_k(y_true: Any, y_score: Any, fraction: float) -> float:
    """Precision at ``k`` divided by the base rate.

    ``1.0`` means the ranking is no better than calling people at random — the
    only honest reference point for a targeting model.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    base_rate = float(true_array.mean())
    if base_rate <= 0.0:
        return float("nan")
    return precision_at_k(true_array, score_array, fraction) / base_rate


def expected_calibration_error(y_true: Any, y_score: Any, *, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Uses equal-count bins so that each bin carries a comparable amount of
    evidence; equal-width bins would be almost empty at the top of the range,
    where this dataset's models place very few rows.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    if true_array.size < n_bins:
        return float("nan")
    quantiles = np.quantile(score_array, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        return float(abs(score_array.mean() - true_array.mean()))
    bins = np.clip(np.digitize(score_array, edges[1:-1], right=True), 0, edges.size - 2)

    total = 0.0
    for index in range(edges.size - 1):
        mask = bins == index
        count = int(mask.sum())
        if count == 0:
            continue
        total += count * abs(score_array[mask].mean() - true_array[mask].mean())
    return float(total / true_array.size)


def calibration_summary(y_true: Any, y_score: Any, *, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin predicted vs observed frequency, for the calibration plot."""
    true_array, score_array = _as_arrays(y_true, y_score)
    frame = pd.DataFrame({"y_true": true_array, "y_score": score_array})
    frame["bin"] = pd.qcut(frame["y_score"], q=n_bins, duplicates="drop", labels=False)
    grouped = frame.groupby("bin", observed=True).agg(
        n=("y_true", "size"),
        mean_predicted=("y_score", "mean"),
        observed_rate=("y_true", "mean"),
    )
    return grouped.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Threshold-free and threshold-dependent metrics for one model on one split.

    Attributes:
        n_rows: Rows scored.
        n_positive: Positive labels present.
        base_rate: Share of positives — the reference every lift is relative to.
        roc_auc: Area under the ROC curve.
        average_precision: Area under the precision-recall curve. Primary metric.
        log_loss: Negative log likelihood, sensitive to calibration.
        brier_score: Mean squared probability error.
        expected_calibration_error: Mean absolute gap, equal-count bins.
        threshold: Decision threshold these point metrics were computed at.
        precision / recall / f1: Positive-class point metrics at ``threshold``.
        confusion: ``tn``, ``fp``, ``fn``, ``tp`` at ``threshold``.
        top_k: ``precision@k``, ``recall@k`` and ``lift@k`` per capacity fraction.
    """

    n_rows: int
    n_positive: int
    base_rate: float
    roc_auc: float
    average_precision: float
    log_loss: float
    brier_score: float
    expected_calibration_error: float
    threshold: float
    precision: float
    recall: float
    f1: float
    confusion: dict[str, int]
    top_k: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON reports."""
        return asdict(self)

    def headline(self) -> str:
        """One-line summary for logs."""
        return (
            f"AP={self.average_precision:.4f} ROC-AUC={self.roc_auc:.4f} "
            f"Brier={self.brier_score:.4f} "
            f"P@20%={self.top_k.get('0.20', {}).get('precision', float('nan')):.4f} "
            f"lift@20%={self.top_k.get('0.20', {}).get('lift', float('nan')):.2f}"
        )


def binary_metrics(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float = 0.5,
    top_k_fractions: Sequence[float] = DEFAULT_TOP_K,
) -> ClassificationMetrics:
    """Compute the full metric set for one set of scores.

    Args:
        y_true: Binary labels.
        y_score: Positive-class probabilities.
        threshold: Cut-off for the point metrics. Chosen on validation data by
            :func:`~term_deposit.evaluation.thresholds.select_threshold`, never
            tuned on the split being reported.
        top_k_fractions: Capacity fractions for lift/precision/recall at k.

    Returns:
        A :class:`ClassificationMetrics`. Ranking metrics become ``nan`` when the
        split contains a single class, rather than raising — a degenerate backtest
        fold should not abort the run.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    n_positive = int(true_array.sum())
    single_class = n_positive == 0 or n_positive == true_array.size

    if single_class:
        logger.warning("Split contains a single class; ranking metrics are undefined")
        roc = average_precision = float("nan")
    else:
        roc = float(roc_auc_score(true_array, score_array))
        average_precision = float(average_precision_score(true_array, score_array))

    predictions = (score_array >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_array, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(true_array, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    clipped = np.clip(score_array, 1e-15, 1 - 1e-15)
    return ClassificationMetrics(
        n_rows=int(true_array.size),
        n_positive=n_positive,
        base_rate=float(true_array.mean()),
        roc_auc=roc,
        average_precision=average_precision,
        log_loss=(
            float(log_loss(true_array, clipped, labels=[0, 1]))
            if not single_class
            else float("nan")
        ),
        brier_score=float(brier_score_loss(true_array, score_array)),
        expected_calibration_error=expected_calibration_error(true_array, score_array),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        top_k={
            f"{fraction:.2f}": {
                "precision": precision_at_k(true_array, score_array, fraction),
                "recall": recall_at_k(true_array, score_array, fraction),
                "lift": lift_at_k(true_array, score_array, fraction),
            }
            for fraction in top_k_fractions
        },
    )


def within_period_metrics(
    y_true: Any,
    y_score: Any,
    periods: Any,
    *,
    min_rows: int = 50,
) -> dict[str, Any]:
    """Ranking quality *inside* each calendar period, and the pooled comparison.

    This is the project's key diagnostic. A pooled ROC-AUC far above the
    within-period average means the model is separating calendar months rather
    than customers — and in production every scored batch belongs to a single
    month, so only the within-period number survives deployment.

    Args:
        y_true: Binary labels.
        y_score: Positive-class probabilities.
        periods: Calendar period per row.
        min_rows: Periods smaller than this are skipped as too noisy to score.

    Returns:
        A mapping with the row-weighted within-period ``roc_auc`` and
        ``average_precision``, the pooled values, the resulting gap, and a
        per-period breakdown.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    period_values = np.asarray(periods).ravel().astype(str)
    if period_values.shape != true_array.shape:
        msg = f"periods must align with labels: {period_values.shape} vs {true_array.shape}"
        raise ValueError(msg)

    breakdown: list[dict[str, Any]] = []
    for period in sorted(set(period_values)):
        mask = period_values == period
        labels = true_array[mask]
        scores = score_array[mask]
        if labels.size < min_rows or labels.min() == labels.max():
            continue
        breakdown.append(
            {
                "period": period,
                "n_rows": int(labels.size),
                "base_rate": float(labels.mean()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "lift_at_20pct": lift_at_k(labels, scores, 0.20),
                "mean_score": float(scores.mean()),
            }
        )

    if not breakdown:
        logger.warning("No calendar period had enough rows for a within-period metric")
        return {
            "n_periods_scored": 0,
            "weighted_roc_auc": float("nan"),
            "weighted_average_precision": float("nan"),
            "weighted_lift_at_20pct": float("nan"),
            "pooled_roc_auc": float("nan"),
            "pooled_average_precision": float("nan"),
            "roc_auc_inflation": float("nan"),
            "by_period": [],
        }

    weights = np.array([entry["n_rows"] for entry in breakdown], dtype=float)
    weighted_roc = float(np.average([e["roc_auc"] for e in breakdown], weights=weights))
    weighted_ap = float(np.average([e["average_precision"] for e in breakdown], weights=weights))
    weighted_lift = float(np.average([e["lift_at_20pct"] for e in breakdown], weights=weights))

    single_class = true_array.min() == true_array.max()
    pooled_roc = float("nan") if single_class else float(roc_auc_score(true_array, score_array))
    pooled_ap = (
        float("nan") if single_class else float(average_precision_score(true_array, score_array))
    )

    return {
        "n_periods_scored": len(breakdown),
        "weighted_roc_auc": weighted_roc,
        "weighted_average_precision": weighted_ap,
        "weighted_lift_at_20pct": weighted_lift,
        "pooled_roc_auc": pooled_roc,
        "pooled_average_precision": pooled_ap,
        # How much of the pooled score comes from telling calendar months apart.
        "roc_auc_inflation": pooled_roc - weighted_roc,
        "by_period": breakdown,
    }


def bootstrap_interval(
    y_true: Any,
    y_score: Any,
    metric: str = "average_precision",
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap interval for a ranking metric.

    Reported so that model comparisons are read against sampling noise. On the
    out-of-time test window (about 2,000 rows) the intervals are wide enough that
    most of the model ranking in the original notebook is not distinguishable.

    Args:
        y_true: Binary labels.
        y_score: Positive-class probabilities.
        metric: ``"average_precision"`` or ``"roc_auc"``.
        n_resamples: Bootstrap resamples. ``0`` disables and returns ``nan`` bounds.
        confidence: Two-sided coverage.
        seed: Seed for the resampling generator.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    scorers = {"average_precision": average_precision_score, "roc_auc": roc_auc_score}
    if metric not in scorers:
        msg = f"unsupported metric {metric!r}; choose from {sorted(scorers)}"
        raise ValueError(msg)
    scorer = scorers[metric]

    point = (
        float("nan")
        if true_array.min() == true_array.max()
        else float(scorer(true_array, score_array))
    )
    if n_resamples <= 0:
        return {"point": point, "lower": float("nan"), "upper": float("nan")}

    rng = make_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, true_array.size, true_array.size)
        resampled = true_array[indices]
        if resampled.min() == resampled.max():
            continue
        samples.append(float(scorer(resampled, score_array[indices])))

    if not samples:
        return {"point": point, "lower": float("nan"), "upper": float("nan")}

    alpha = (1.0 - confidence) / 2.0
    return {
        "point": point,
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
    }


def metrics_to_row(name: str, metrics: ClassificationMetrics, **extra: Any) -> dict[str, Any]:
    """Flatten metrics into one record for a comparison table."""
    row: dict[str, Any] = {
        "model": name,
        "n_rows": metrics.n_rows,
        "base_rate": metrics.base_rate,
        "average_precision": metrics.average_precision,
        "roc_auc": metrics.roc_auc,
        "brier_score": metrics.brier_score,
        "ece": metrics.expected_calibration_error,
        "threshold": metrics.threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }
    for fraction, values in metrics.top_k.items():
        row[f"precision_at_{fraction}"] = values["precision"]
        row[f"lift_at_{fraction}"] = values["lift"]
    row.update(extra)
    return row


def comparison_frame(rows: Sequence[Mapping[str, Any]], sort_by: str) -> pd.DataFrame:
    """Assemble metric rows into a table sorted by the primary metric."""
    frame = pd.DataFrame(list(rows))
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=False, kind="stable")
    return frame.reset_index(drop=True)
