"""Choosing the decision threshold.

The original notebook compared four models at a fixed 0.5 cut-off while three of
them carried class weights and one did not. That comparison measures how each
estimator's probability scale was rescaled by its balancing strategy, not how
well it ranks customers: the "high recall, low precision" and "high precision,
low recall" groupings in the results table line up exactly with which models were
weighted.

Here the threshold is a decision variable, selected on the *validation* split
under an explicit objective and then applied unchanged to the test split. Every
model is then compared at the operating point it was tuned for, and a threshold
sweep is reported so the trade-off is visible rather than implied. See ADR 0005.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from term_deposit.evaluation.metrics import _as_arrays, _rank_order, top_k_count
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

ThresholdObjective = Literal["f1", "expected_value", "top_k"]


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """A selected operating point and the evidence behind it.

    Attributes:
        threshold: The probability cut-off.
        objective: Rule used to choose it.
        objective_value: Value of that objective on the split it was chosen from.
        selected_fraction: Share of the list the threshold marks for calling.
        expected_value_per_contact: Net value per *scored* customer at this point.
        chosen_on: Which split the choice was made on. Always ``"validation"`` in
            the pipeline; recorded so a report can never imply otherwise.
    """

    threshold: float
    objective: str
    objective_value: float
    selected_fraction: float
    expected_value_per_contact: float
    chosen_on: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON reports."""
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "selected_fraction": self.selected_fraction,
            "expected_value_per_contact": self.expected_value_per_contact,
            "chosen_on": self.chosen_on,
        }


def expected_value(
    y_true: Any,
    y_score: Any,
    threshold: float,
    *,
    value_per_subscription: float,
    cost_per_call: float,
) -> float:
    """Net value per scored customer if everyone above ``threshold`` is called.

    ``value_per_subscription`` and ``cost_per_call`` are placeholders in
    consistent units, not measured euros. They make the trade-off between a
    missed subscriber and a wasted call *explicit and adjustable* instead of
    hidden inside a default 0.5 cut-off; the numbers a bank plugs in would come
    from its own margin and cost accounting. See ``docs/methodology.md``.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    called = score_array >= threshold
    n_called = int(called.sum())
    if n_called == 0:
        return 0.0
    conversions = float(true_array[called].sum())
    return (conversions * value_per_subscription - n_called * cost_per_call) / true_array.size


def threshold_sweep(
    y_true: Any,
    y_score: Any,
    *,
    value_per_subscription: float = 100.0,
    cost_per_call: float = 5.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Tabulate precision, recall, F1 and expected value across thresholds.

    Candidate thresholds are score quantiles rather than an even grid on ``[0, 1]``.
    Class-weighted models concentrate their scores in a narrow band, and an even
    grid would spend most of its points in a region containing no data.

    Returns:
        One row per candidate threshold, ascending.
    """
    true_array, score_array = _as_arrays(y_true, y_score)
    candidates = np.unique(
        np.quantile(score_array, np.linspace(0.0, 1.0, n_thresholds, endpoint=False))
    )
    positives = float(true_array.sum())

    records = []
    for threshold in candidates:
        called = score_array >= threshold
        n_called = int(called.sum())
        if n_called == 0:
            continue
        true_positives = float(true_array[called].sum())
        precision = true_positives / n_called
        recall = true_positives / positives if positives else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        records.append(
            {
                "threshold": float(threshold),
                "selected_fraction": n_called / true_array.size,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "expected_value_per_contact": (
                    true_positives * value_per_subscription - n_called * cost_per_call
                )
                / true_array.size,
            }
        )
    return pd.DataFrame.from_records(records)


def select_threshold(
    y_true: Any,
    y_score: Any,
    *,
    objective: ThresholdObjective = "expected_value",
    value_per_subscription: float = 100.0,
    cost_per_call: float = 5.0,
    capacity_fraction: float = 0.20,
    chosen_on: str = "validation",
) -> ThresholdChoice:
    """Pick a decision threshold under an explicit objective.

    Objectives:

    ``expected_value``
        Maximise net value per scored customer. The economically meaningful
        default: it encodes what a false negative costs relative to a wasted call.
    ``f1``
        Maximise positive-class F1. Assumes precision and recall matter equally,
        which is a claim about the business, not a property of the data.
    ``top_k``
        Call exactly ``capacity_fraction`` of the list. The right choice when the
        campaign's constraint is agent hours rather than a probability cut-off.

    Args:
        y_true: Binary labels on the selection split.
        y_score: Positive-class probabilities on the selection split.
        objective: Selection rule.
        value_per_subscription: Value of one conversion.
        cost_per_call: Cost of one call attempt.
        capacity_fraction: List share callable, used by ``top_k``.
        chosen_on: Name of the split, recorded for provenance.

    Returns:
        A :class:`ThresholdChoice`.

    Raises:
        ValueError: If the objective is unknown or the split has no usable rows.
    """
    true_array, score_array = _as_arrays(y_true, y_score)

    valid = ("expected_value", "f1", "top_k")
    if objective not in valid:
        msg = f"unknown threshold objective {objective!r}; choose from {list(valid)}"
        raise ValueError(msg)

    if objective == "top_k":
        k = top_k_count(true_array.size, capacity_fraction)
        ordered_scores = score_array[_rank_order(score_array)]
        threshold = float(ordered_scores[k - 1])
    else:
        sweep = threshold_sweep(
            true_array,
            score_array,
            value_per_subscription=value_per_subscription,
            cost_per_call=cost_per_call,
        )
        if sweep.empty:
            msg = "threshold sweep produced no candidates; check the score distribution"
            raise ValueError(msg)
        column = "f1" if objective == "f1" else "expected_value_per_contact"
        threshold = float(sweep["threshold"].iloc[int(sweep[column].idxmax())])

    called = score_array >= threshold
    selected_fraction = float(called.mean())
    net_value = expected_value(
        true_array,
        score_array,
        threshold,
        value_per_subscription=value_per_subscription,
        cost_per_call=cost_per_call,
    )

    if objective == "f1":
        true_positives = float(true_array[called].sum())
        precision = true_positives / max(1, int(called.sum()))
        recall = true_positives / max(1.0, float(true_array.sum()))
        objective_value = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
    else:
        objective_value = net_value

    logger.debug(
        "Selected threshold %.4f on %s (objective=%s, calls %.1f%% of the list)",
        threshold,
        chosen_on,
        objective,
        selected_fraction * 100,
    )
    return ThresholdChoice(
        threshold=threshold,
        objective=objective,
        objective_value=float(objective_value),
        selected_fraction=selected_fraction,
        expected_value_per_contact=net_value,
        chosen_on=chosen_on,
    )


def assign_tiers(
    scores: Any,
    *,
    quantiles: tuple[float, ...],
    labels: tuple[str, ...],
) -> np.ndarray:
    """Bucket scores into priority tiers by rank.

    Tiers are assigned on *rank*, not on absolute probability, so the tier
    definition survives the recalibration that a regime change forces. Tier 1 is
    the highest-scoring ``quantiles[0]`` share of the batch.

    Args:
        scores: Positive-class probabilities.
        quantiles: Ascending cumulative shares marking each tier boundary.
        labels: One more label than there are boundaries.

    Returns:
        An array of tier labels aligned with ``scores``.
    """
    if len(labels) != len(quantiles) + 1:
        msg = f"expected {len(quantiles) + 1} labels, got {len(labels)}"
        raise ValueError(msg)

    score_array = np.asarray(scores, dtype=float).ravel()
    if score_array.size == 0:
        return np.array([], dtype=object)

    # Rank 0 is the highest score; ties resolve by position for determinism.
    ranks = np.empty(score_array.size, dtype=int)
    ranks[_rank_order(score_array)] = np.arange(score_array.size)
    positions = ranks / score_array.size

    tiers = np.full(score_array.size, labels[-1], dtype=object)
    for boundary, label in zip(reversed(quantiles), reversed(labels[:-1]), strict=True):
        tiers[positions < boundary] = label
    return tiers
