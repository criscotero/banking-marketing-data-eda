"""Metrics.

Each metric is checked against a case where the correct answer is known by
construction — a perfect ranking, a constant score, a single class — rather than
against a golden number produced by the code itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from term_deposit.evaluation.metrics import (
    binary_metrics,
    bootstrap_interval,
    calibration_summary,
    expected_calibration_error,
    lift_at_k,
    precision_at_k,
    recall_at_k,
    top_k_count,
    within_period_metrics,
)


@pytest.fixture
def perfect_ranking() -> tuple[np.ndarray, np.ndarray]:
    """20 positives at the top of a 100-row list, scored perfectly."""
    labels = np.array([1] * 20 + [0] * 80)
    scores = np.linspace(1.0, 0.0, 100)
    return labels, scores


class TestTopKCount:
    def test_rounds_to_the_nearest_row(self):
        assert top_k_count(100, 0.20) == 20
        assert top_k_count(37, 0.10) == 4

    def test_always_returns_at_least_one_row(self):
        assert top_k_count(3, 0.01) == 1

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_rejects_fractions_outside_the_unit_interval(self, fraction):
        with pytest.raises(ValueError, match="fraction must be"):
            top_k_count(100, fraction)


class TestPrecisionRecallAtK:
    def test_perfect_ranking_captures_every_positive_in_the_top_20_percent(self, perfect_ranking):
        labels, scores = perfect_ranking
        assert precision_at_k(labels, scores, 0.20) == pytest.approx(1.0)
        assert recall_at_k(labels, scores, 0.20) == pytest.approx(1.0)

    def test_lift_equals_the_inverse_base_rate_for_a_perfect_ranking(self, perfect_ranking):
        labels, scores = perfect_ranking
        assert lift_at_k(labels, scores, 0.20) == pytest.approx(1.0 / 0.20)

    def test_constant_scores_give_a_lift_near_one(self):
        """A model with no signal must not score a lift above 1.

        Ties are broken by a fixed random permutation. Breaking them on row order
        instead would rank by contact date, which in this dataset is itself a
        signal — the failure this test exists to prevent.
        """
        labels = np.array([1] * 200 + [0] * 800)  # sorted, worst case for stable sort
        constant = np.full(1000, 0.5)
        assert lift_at_k(labels, constant, 0.20) == pytest.approx(1.0, abs=0.25)

    def test_tie_breaking_is_reproducible(self):
        labels = np.array([1] * 50 + [0] * 50)
        constant = np.full(100, 0.5)
        assert lift_at_k(labels, constant, 0.3) == lift_at_k(labels, constant, 0.3)

    def test_recall_is_undefined_without_positives(self):
        assert np.isnan(recall_at_k(np.zeros(10), np.linspace(0, 1, 10), 0.2))

    def test_lift_is_undefined_without_positives(self):
        assert np.isnan(lift_at_k(np.zeros(10), np.linspace(0, 1, 10), 0.2))


class TestBinaryMetrics:
    def test_perfect_scores_give_perfect_ranking_metrics(self, perfect_ranking):
        labels, scores = perfect_ranking
        metrics = binary_metrics(labels, scores, threshold=0.5)
        assert metrics.roc_auc == pytest.approx(1.0)
        assert metrics.average_precision == pytest.approx(1.0)

    def test_random_scores_give_a_roc_auc_near_one_half(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 4000)
        metrics = binary_metrics(labels, rng.random(4000))
        assert metrics.roc_auc == pytest.approx(0.5, abs=0.05)

    def test_no_skill_average_precision_equals_the_base_rate(self):
        """The reference point for average precision.

        Any AP at or below the base rate is worthless, which is why the
        comparison table always reports the base rate alongside it.
        """
        rng = np.random.default_rng(1)
        labels = (rng.random(8000) < 0.11).astype(int)
        metrics = binary_metrics(labels, rng.random(8000))
        assert metrics.average_precision == pytest.approx(metrics.base_rate, abs=0.02)

    def test_confusion_counts_sum_to_the_row_count(self, perfect_ranking):
        labels, scores = perfect_ranking
        metrics = binary_metrics(labels, scores, threshold=0.5)
        assert sum(metrics.confusion.values()) == metrics.n_rows

    def test_threshold_is_applied_inclusively(self):
        metrics = binary_metrics(np.array([0, 1]), np.array([0.3, 0.7]), threshold=0.7)
        assert metrics.confusion["tp"] == 1
        assert metrics.confusion["fp"] == 0

    def test_a_single_class_split_yields_nan_ranking_metrics_without_raising(self):
        """A degenerate backtest fold must not abort a whole run."""
        metrics = binary_metrics(np.zeros(50), np.linspace(0, 1, 50))
        assert np.isnan(metrics.roc_auc)
        assert np.isnan(metrics.average_precision)
        assert metrics.n_rows == 50

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError, match="must align"):
            binary_metrics(np.zeros(5), np.zeros(6))

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="empty"):
            binary_metrics(np.array([]), np.array([]))

    def test_reports_every_requested_capacity_fraction(self, perfect_ranking):
        labels, scores = perfect_ranking
        metrics = binary_metrics(labels, scores, top_k_fractions=(0.1, 0.25))
        assert set(metrics.top_k) == {"0.10", "0.25"}


class TestCalibration:
    def test_perfectly_calibrated_scores_have_near_zero_error(self):
        rng = np.random.default_rng(3)
        probabilities = rng.uniform(0.05, 0.95, 20_000)
        labels = (rng.random(20_000) < probabilities).astype(int)
        assert expected_calibration_error(labels, probabilities) < 0.02

    def test_systematically_inflated_scores_are_detected(self):
        """The failure mode class weighting introduces: correct ranking, wrong scale."""
        rng = np.random.default_rng(4)
        true_probability = rng.uniform(0.02, 0.20, 20_000)
        labels = (rng.random(20_000) < true_probability).astype(int)
        inflated = np.clip(true_probability * 4.0, 0, 1)
        assert expected_calibration_error(labels, inflated) > 0.10

    def test_summary_bins_are_ordered_by_predicted_probability(self):
        rng = np.random.default_rng(5)
        scores = rng.random(1000)
        labels = (rng.random(1000) < scores).astype(int)
        summary = calibration_summary(labels, scores, n_bins=10)
        assert summary["mean_predicted"].is_monotonic_increasing
        assert summary["n"].sum() == 1000


class TestWithinPeriodMetrics:
    def test_detects_a_score_that_only_separates_periods(self):
        """The project's core diagnostic, on a case built to isolate it.

        Scores encode the period and nothing else: pooled ROC-AUC is near
        perfect, within-period ROC-AUC is exactly chance.
        """
        rng = np.random.default_rng(6)
        labels: list[int] = []
        scores: list[float] = []
        periods: list[str] = []
        for index, rate in enumerate([0.05, 0.25, 0.60]):
            n = 400
            labels.extend((rng.random(n) < rate).astype(int))
            scores.extend(np.full(n, rate) + rng.normal(0, 1e-6, n))
            periods.extend([f"2009-{index + 1:02d}"] * n)

        result = within_period_metrics(np.array(labels), np.array(scores), np.array(periods))
        assert result["pooled_roc_auc"] > 0.75
        assert result["weighted_roc_auc"] == pytest.approx(0.5, abs=0.05)
        assert result["roc_auc_inflation"] > 0.25

    def test_a_genuine_within_period_signal_survives(self):
        rng = np.random.default_rng(7)
        labels: list[int] = []
        scores: list[float] = []
        periods: list[str] = []
        for index in range(3):
            n = 400
            score = rng.random(n)
            labels.extend((rng.random(n) < score).astype(int))
            scores.extend(score)
            periods.extend([f"2009-{index + 1:02d}"] * n)
        result = within_period_metrics(np.array(labels), np.array(scores), np.array(periods))
        assert result["weighted_roc_auc"] > 0.7
        assert abs(result["roc_auc_inflation"]) < 0.1

    def test_skips_periods_below_the_row_threshold(self):
        labels = np.array([0, 1] * 50)
        scores = np.linspace(0, 1, 100)
        periods = np.array(["2009-01"] * 90 + ["2009-02"] * 10)
        result = within_period_metrics(labels, scores, periods, min_rows=50)
        assert result["n_periods_scored"] == 1

    def test_returns_nan_when_no_period_qualifies(self):
        result = within_period_metrics(
            np.array([0, 1] * 10), np.linspace(0, 1, 20), np.array(["2009-01"] * 20), min_rows=1000
        )
        assert result["n_periods_scored"] == 0
        assert np.isnan(result["weighted_roc_auc"])

    def test_rejects_misaligned_periods(self):
        with pytest.raises(ValueError, match="must align"):
            within_period_metrics(np.zeros(5), np.zeros(5), np.array(["a", "b"]))


class TestBootstrapInterval:
    def test_the_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(8)
        labels = (rng.random(1500) < 0.2).astype(int)
        scores = labels * 0.4 + rng.random(1500) * 0.6
        interval = bootstrap_interval(labels, scores, n_resamples=200, seed=1)
        assert interval["lower"] <= interval["point"] <= interval["upper"]

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(9)
        labels = (rng.random(600) < 0.3).astype(int)
        scores = rng.random(600)
        first = bootstrap_interval(labels, scores, n_resamples=100, seed=3)
        second = bootstrap_interval(labels, scores, n_resamples=100, seed=3)
        assert first == second

    def test_zero_resamples_disables_the_interval(self):
        labels = np.array([0, 1] * 50)
        interval = bootstrap_interval(labels, np.linspace(0, 1, 100), n_resamples=0)
        assert np.isnan(interval["lower"])
        assert not np.isnan(interval["point"])

    def test_rejects_an_unsupported_metric(self):
        with pytest.raises(ValueError, match="unsupported metric"):
            bootstrap_interval(np.array([0, 1]), np.array([0.1, 0.9]), metric="accuracy")
