"""Split construction.

The tests that matter here are the ones that would catch contamination: no row in
two partitions, no future row in training, and no label-derived quantity computed
outside the training split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from term_deposit import constants
from term_deposit.config import SplitConfig
from term_deposit.data.splits import (
    DataSplit,
    SplitError,
    SplitIndices,
    describe_drift,
    make_split,
    rolling_origin_folds,
    stratified_group_check,
)

FEATURES = ("age", "campaign", "pdays", "previous", "job", "poutcome", "euribor3m")


def _split(frame: pd.DataFrame, **overrides: object) -> DataSplit:
    config = SplitConfig(
        **{"strategy": "out_of_time", "test_periods": 3, "validation_periods": 3, **overrides}
    )
    return make_split(frame, config, feature_columns=FEATURES)


class TestSplitIndices:
    def test_rejects_overlapping_partitions(self):
        shared = pd.Index([1, 2, 3])
        with pytest.raises(SplitError, match="overlap"):
            SplitIndices(train=shared, validation=pd.Index([3, 4]), test=pd.Index([5]))

    def test_accepts_disjoint_partitions(self):
        indices = SplitIndices(train=pd.Index([1, 2]), validation=pd.Index([3]), test=pd.Index([4]))
        assert indices.sizes == {"train": 2, "validation": 1, "test": 1}


class TestOutOfTimeSplit:
    def test_partitions_are_disjoint_and_exhaustive(self, labelled_frame):
        split = _split(labelled_frame)
        total = sum(split.sizes.values())
        assert total == len(labelled_frame)
        assert not set(split.X_train.index) & set(split.X_test.index)

    def test_every_training_row_precedes_every_test_row(self, labelled_frame):
        """The property that makes the protocol out-of-time.

        A single leaked future row would let the model see the regime it is
        being asked to generalise to.
        """
        split = _split(labelled_frame)
        assert split.periods_for("train").max() < split.periods_for("test").min()
        assert split.periods_for("validation").max() < split.periods_for("test").min()
        assert split.periods_for("train").max() < split.periods_for("validation").min()

    def test_records_the_calendar_boundaries(self, labelled_frame):
        split = _split(labelled_frame)
        assert set(split.boundaries) == {
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        }

    def test_test_window_holds_exactly_the_configured_number_of_periods(self, labelled_frame):
        split = _split(labelled_frame, test_periods=4)
        assert split.periods_for("test").nunique() == 4

    def test_fails_when_the_data_has_too_few_periods(self, labelled_frame):
        with pytest.raises(SplitError, match="at least"):
            _split(labelled_frame, test_periods=40, validation_periods=10)

    def test_base_rate_drifts_between_train_and_test(self, labelled_frame):
        """Confirms the fixture reproduces the real dataset's regime shift.

        If this stopped holding, the split tests below would still pass while
        silently testing a much easier problem than the real one.
        """
        split = _split(labelled_frame)
        rates = split.positive_rates
        assert rates["test"] > rates["train"]


class TestRandomSplit:
    def test_partitions_are_disjoint(self, labelled_frame):
        split = _split(labelled_frame, strategy="random", test_size=0.2, validation_size=0.15)
        assert not set(split.X_train.index) & set(split.X_test.index)
        assert not set(split.X_validation.index) & set(split.X_test.index)

    def test_fractions_are_relative_to_the_whole_dataset(self, labelled_frame):
        split = _split(labelled_frame, strategy="random", test_size=0.2, validation_size=0.15)
        total = len(labelled_frame)
        assert split.sizes["test"] / total == pytest.approx(0.2, abs=0.01)
        assert split.sizes["validation"] / total == pytest.approx(0.15, abs=0.01)

    def test_stratification_preserves_the_base_rate(self, labelled_frame):
        split = _split(labelled_frame, strategy="random", test_size=0.2, validation_size=0.15)
        overall = float(labelled_frame[constants.LABEL_COLUMN].mean())
        for rate in split.positive_rates.values():
            assert rate == pytest.approx(overall, abs=0.03)

    def test_is_reproducible_for_a_fixed_seed(self, labelled_frame):
        first = _split(labelled_frame, strategy="random", random_state=1)
        second = _split(labelled_frame, strategy="random", random_state=1)
        assert list(first.X_test.index) == list(second.X_test.index)

    def test_a_different_seed_produces_a_different_split(self, labelled_frame):
        first = _split(labelled_frame, strategy="random", random_state=1)
        second = _split(labelled_frame, strategy="random", random_state=2)
        assert list(first.X_test.index) != list(second.X_test.index)

    def test_rejects_fractions_that_leave_no_training_rows(self):
        with pytest.raises(ValueError, match="leave room"):
            SplitConfig(strategy="random", test_size=0.6, validation_size=0.5)


class TestScalePosWeight:
    def test_is_derived_from_the_training_split_only(self, labelled_frame):
        """Derived from training rows only.

        Computing it from the full dataset would let the test set's class
        balance influence a training hyperparameter.
        """
        split = _split(labelled_frame)
        rate = float(split.y_train.mean())
        assert split.scale_pos_weight() == pytest.approx((1 - rate) / rate)

    def test_rejects_a_single_class_training_split(self, labelled_frame):
        split = _split(labelled_frame)
        degenerate = DataSplit(
            strategy=split.strategy,
            X_train=split.X_train,
            y_train=split.y_train * 0,
            X_validation=split.X_validation,
            y_validation=split.y_validation,
            X_test=split.X_test,
            y_test=split.y_test,
            periods=split.periods,
            boundaries=split.boundaries,
        )
        with pytest.raises(SplitError, match="degenerate"):
            degenerate.scale_pos_weight()


class TestMakeSplitValidation:
    def test_requires_the_label_column(self, labelled_frame):
        with pytest.raises(SplitError, match="required"):
            _split(labelled_frame.drop(columns=[constants.LABEL_COLUMN]))

    def test_requires_the_period_column(self, labelled_frame):
        with pytest.raises(SplitError, match="required"):
            _split(labelled_frame.drop(columns=[constants.PERIOD_COLUMN]))

    def test_reports_missing_feature_columns(self, labelled_frame):
        config = SplitConfig(strategy="out_of_time", test_periods=3, validation_periods=3)
        with pytest.raises(SplitError, match="not present"):
            make_split(labelled_frame, config, feature_columns=("age", "does_not_exist"))


class TestRollingOriginFolds:
    def test_training_rows_always_precede_the_test_period(self, labelled_frame):
        periods = labelled_frame[constants.PERIOD_COLUMN]
        folds = rolling_origin_folds(periods, n_folds=4, min_test_rows=10)
        assert folds
        for train_index, test_index, label in folds:
            assert periods.loc[train_index].max() < periods.loc[test_index].min()
            assert set(periods.loc[test_index].astype(str)) == {label}

    def test_the_training_window_expands(self, labelled_frame):
        periods = labelled_frame[constants.PERIOD_COLUMN]
        folds = rolling_origin_folds(periods, n_folds=4, min_test_rows=10)
        sizes = [len(train_index) for train_index, _, _ in folds]
        assert sizes == sorted(sizes)

    def test_skips_periods_below_the_row_threshold(self, labelled_frame):
        periods = labelled_frame[constants.PERIOD_COLUMN]
        assert rolling_origin_folds(periods, n_folds=4, min_test_rows=10_000) == []


class TestDiagnostics:
    def test_drift_ranks_macro_features_highest(self, labelled_frame):
        """The macro block shifts hardest between train and test.

        Under an out-of-time split it is a proxy for the calendar, so a change
        of window moves it further than any client attribute.
        """
        split = _split(labelled_frame)
        drift = describe_drift(split)
        top = set(drift.head(3)["feature"])
        assert top & set(constants.MACRO_FEATURES)

    def test_stratified_group_check_reports_one_row_per_period(self, labelled_frame):
        check = stratified_group_check(
            labelled_frame[constants.LABEL_COLUMN], labelled_frame[constants.PERIOD_COLUMN]
        )
        assert len(check) == labelled_frame[constants.PERIOD_COLUMN].nunique()
        assert check["positive_rate"].between(0.0, 1.0).all()
