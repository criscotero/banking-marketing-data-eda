"""Threshold selection and tiering.

The property that matters most is provenance: the threshold must be chosen on
data the model is not being scored on. These tests pin that, and pin the
behaviour that made the original notebook's model comparison unsound — comparing
differently-weighted models at a shared, arbitrary 0.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from term_deposit.evaluation.thresholds import (
    assign_tiers,
    expected_value,
    select_threshold,
    threshold_sweep,
)


@pytest.fixture
def signal() -> tuple[np.ndarray, np.ndarray]:
    """2,000 rows with a 20% base rate and a genuine but imperfect signal."""
    rng = np.random.default_rng(11)
    labels = (rng.random(2000) < 0.2).astype(int)
    scores = np.clip(labels * 0.35 + rng.normal(0.3, 0.2, 2000), 0.0, 1.0)
    return labels, scores


class TestExpectedValue:
    def test_calling_nobody_is_worth_nothing(self, signal):
        labels, scores = signal
        value = expected_value(
            labels, scores, threshold=1.1, value_per_subscription=100, cost_per_call=5
        )
        assert value == 0.0

    def test_calling_everyone_equals_base_rate_value_minus_cost(self, signal):
        labels, scores = signal
        value = expected_value(
            labels, scores, threshold=0.0, value_per_subscription=100, cost_per_call=5
        )
        assert value == pytest.approx(labels.mean() * 100 - 5)

    def test_a_free_call_makes_calling_everyone_optimal(self, signal):
        labels, scores = signal
        call_all = expected_value(
            labels, scores, threshold=0.0, value_per_subscription=100, cost_per_call=0
        )
        selective = expected_value(
            labels, scores, threshold=0.5, value_per_subscription=100, cost_per_call=0
        )
        assert call_all >= selective


class TestThresholdSweep:
    def test_recall_falls_as_the_threshold_rises(self, signal):
        labels, scores = signal
        sweep = threshold_sweep(labels, scores).sort_values("threshold")
        assert sweep["recall"].is_monotonic_decreasing

    def test_the_selected_fraction_falls_as_the_threshold_rises(self, signal):
        labels, scores = signal
        sweep = threshold_sweep(labels, scores).sort_values("threshold")
        assert sweep["selected_fraction"].is_monotonic_decreasing

    def test_candidates_come_from_the_score_distribution(self):
        """Candidates follow the score distribution.

        An even grid on ``[0, 1]`` would spend most of its points where no
        scores live, which is exactly what happens with class-weighted models.
        """
        labels = np.array([0, 1] * 500)
        narrow = np.random.default_rng(0).uniform(0.80, 0.85, 1000)
        sweep = threshold_sweep(labels, narrow)
        assert sweep["threshold"].min() >= 0.80
        assert sweep["threshold"].max() <= 0.85


class TestSelectThreshold:
    def test_records_which_split_it_was_chosen_on(self, signal):
        """Provenance is the point: a threshold tuned on test is not a threshold."""
        labels, scores = signal
        choice = select_threshold(labels, scores, chosen_on="validation")
        assert choice.chosen_on == "validation"

    def test_expected_value_beats_a_naive_half_threshold(self, signal):
        labels, scores = signal
        choice = select_threshold(
            labels, scores, objective="expected_value", value_per_subscription=100, cost_per_call=5
        )
        naive = expected_value(labels, scores, 0.5, value_per_subscription=100, cost_per_call=5)
        assert choice.expected_value_per_contact >= naive

    def test_f1_objective_maximises_f1(self, signal):
        labels, scores = signal
        choice = select_threshold(labels, scores, objective="f1")
        sweep = threshold_sweep(labels, scores)
        assert choice.objective_value == pytest.approx(sweep["f1"].max(), abs=1e-6)

    def test_top_k_selects_the_configured_capacity(self, signal):
        labels, scores = signal
        choice = select_threshold(labels, scores, objective="top_k", capacity_fraction=0.20)
        assert choice.selected_fraction == pytest.approx(0.20, abs=0.02)

    def test_a_higher_call_cost_raises_the_threshold(self, signal):
        """The economics drive the operating point, not a convention."""
        labels, scores = signal
        cheap = select_threshold(labels, scores, value_per_subscription=100, cost_per_call=1)
        pricey = select_threshold(labels, scores, value_per_subscription=100, cost_per_call=30)
        assert pricey.threshold >= cheap.threshold

    def test_rescaling_scores_does_not_change_who_is_called(self):
        """Per-model thresholds remove the class-weighting confound.

        The original notebook's comparison mixed ranking quality with the
        probability rescaling that class weighting introduces. Selecting a
        threshold per model separates them: a monotone rescaling of the scores
        then selects exactly the same rows.
        """
        rng = np.random.default_rng(12)
        labels = (rng.random(3000) < 0.15).astype(int)
        raw = np.clip(labels * 0.25 + rng.normal(0.2, 0.15, 3000), 1e-6, 1 - 1e-6)
        # Class weighting multiplies the odds by a constant. That is strictly
        # monotone and stays inside (0, 1), so the ranking is untouched and only
        # the probability scale moves.
        weight = 8.0
        inflated = weight * raw / (1.0 + (weight - 1.0) * raw)

        raw_choice = select_threshold(labels, raw, objective="top_k", capacity_fraction=0.2)
        inflated_choice = select_threshold(
            labels, inflated, objective="top_k", capacity_fraction=0.2
        )
        assert set(np.flatnonzero(raw >= raw_choice.threshold)) == set(
            np.flatnonzero(inflated >= inflated_choice.threshold)
        )

    def test_rejects_an_unknown_objective(self, signal):
        labels, scores = signal
        with pytest.raises(ValueError, match="unknown threshold objective"):
            select_threshold(labels, scores, objective="accuracy")  # type: ignore[arg-type]

    def test_serialises_to_a_plain_dict(self, signal):
        labels, scores = signal
        payload = select_threshold(labels, scores).to_dict()
        assert set(payload) == {
            "threshold",
            "objective",
            "objective_value",
            "selected_fraction",
            "expected_value_per_contact",
            "chosen_on",
        }


class TestAssignTiers:
    def test_tier_sizes_follow_the_configured_quantiles(self):
        scores = np.linspace(0, 1, 1000)
        tiers = assign_tiers(scores, quantiles=(0.05, 0.20, 0.50), labels=("t1", "t2", "t3", "t4"))
        counts = {label: int((tiers == label).sum()) for label in ("t1", "t2", "t3", "t4")}
        assert counts == {"t1": 50, "t2": 150, "t3": 300, "t4": 500}

    def test_the_top_tier_holds_the_highest_scores(self):
        scores = np.array([0.1, 0.9, 0.5, 0.99])
        tiers = assign_tiers(scores, quantiles=(0.25,), labels=("top", "rest"))
        assert tiers[3] == "top"
        assert tiers[0] == "rest"

    def test_tiers_are_rank_based_not_probability_based(self):
        """So the tier definition survives recalibration for a new regime."""
        scores = np.linspace(0.01, 0.05, 200)  # all low, but still rankable
        tiers = assign_tiers(scores, quantiles=(0.10,), labels=("top", "rest"))
        assert int((tiers == "top").sum()) == 20

    def test_a_monotone_rescaling_produces_identical_tiers(self):
        rng = np.random.default_rng(13)
        scores = rng.random(500)
        first = assign_tiers(scores, quantiles=(0.2,), labels=("a", "b"))
        second = assign_tiers(scores**2, quantiles=(0.2,), labels=("a", "b"))
        assert (first == second).all()

    def test_rejects_a_label_count_mismatch(self):
        with pytest.raises(ValueError, match="expected 3 labels"):
            assign_tiers(np.array([0.5]), quantiles=(0.1, 0.5), labels=("a", "b"))

    def test_handles_an_empty_batch(self):
        assert assign_tiers(np.array([]), quantiles=(0.1,), labels=("a", "b")).size == 0
