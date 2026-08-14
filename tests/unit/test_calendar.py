"""Calendar reconstruction.

This is the highest-risk piece of logic in the project: the out-of-time split,
the backtest and the within-period metrics all rest on it, and it infers
information (the year) that the raw file does not contain. If it were wrong,
every temporal claim in the README would be wrong with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from term_deposit import constants
from term_deposit.features.calendar import (
    CalendarReconstructionError,
    add_contact_period,
    macro_period_collinearity,
    period_summary,
    reconstruct_contact_period,
)


class TestReconstructContactPeriod:
    def test_assigns_start_year_before_any_rollover(self):
        periods = reconstruct_contact_period(pd.Series(["may", "jun", "jul"]), start_year=2008)
        assert [str(p) for p in periods] == ["2008-05", "2008-06", "2008-07"]

    def test_increments_year_when_the_month_number_decreases(self):
        periods = reconstruct_contact_period(pd.Series(["nov", "dec", "mar"]), start_year=2008)
        assert [str(p) for p in periods] == ["2008-11", "2008-12", "2009-03"]

    def test_handles_several_rollovers(self):
        months = ["may", "dec", "mar", "dec", "mar"]
        periods = reconstruct_contact_period(pd.Series(months), start_year=2008)
        assert [str(p) for p in periods] == [
            "2008-05",
            "2008-12",
            "2009-03",
            "2009-12",
            "2010-03",
        ]

    def test_repeated_months_within_a_block_share_one_period(self):
        periods = reconstruct_contact_period(pd.Series(["may"] * 4 + ["jun"] * 2), start_year=2008)
        assert [str(p) for p in periods] == ["2008-05"] * 4 + ["2008-06"] * 2

    def test_rejects_unknown_month_labels(self):
        with pytest.raises(CalendarReconstructionError, match="unrecognised month"):
            reconstruct_contact_period(pd.Series(["may", "smarch"]))

    def test_rejects_an_empty_column(self):
        with pytest.raises(CalendarReconstructionError, match="empty"):
            reconstruct_contact_period(pd.Series([], dtype=object))

    def test_a_multi_year_campaign_is_accepted(self):
        """Repeating a month label is normal: the campaign ran for three years."""
        periods = reconstruct_contact_period(pd.Series(["may", "jun", "may", "jun"]))
        assert [str(p) for p in periods] == ["2008-05", "2008-06", "2009-05", "2009-06"]

    def test_strict_mode_rejects_shuffled_rows(self):
        """Rows out of contact order would invalidate every temporal result.

        Shuffling makes almost every adjacent pair a month decrease, so the
        reconstruction produces hundreds of "years" instead of three.
        """
        rng = np.random.default_rng(0)
        shuffled = pd.Series(rng.choice(list(constants.MONTH_ABBREVIATIONS), 4000))
        with pytest.raises(CalendarReconstructionError, match="not in contact order"):
            reconstruct_contact_period(shuffled, strict=True)

    def test_non_strict_mode_allows_an_implausible_span(self):
        rng = np.random.default_rng(0)
        shuffled = pd.Series(rng.choice(list(constants.MONTH_ABBREVIATIONS), 4000))
        periods = reconstruct_contact_period(shuffled, strict=False)
        assert periods.nunique() > 120


class TestAddContactPeriod:
    def test_adds_the_period_column_without_mutating_the_input(self, raw_frame):
        before = raw_frame.copy()
        result = add_contact_period(raw_frame)
        assert constants.PERIOD_COLUMN in result.columns
        assert constants.PERIOD_COLUMN not in raw_frame.columns
        pd.testing.assert_frame_equal(raw_frame, before)

    def test_requires_the_month_column(self, raw_frame):
        with pytest.raises(CalendarReconstructionError, match="cannot reconstruct"):
            add_contact_period(raw_frame.drop(columns=["month"]), month_column="month")

    def test_periods_are_monotonically_non_decreasing(self, labelled_frame):
        periods = labelled_frame[constants.PERIOD_COLUMN]
        assert (periods.to_numpy()[1:] >= periods.to_numpy()[:-1]).all()


class TestPeriodSummary:
    def test_one_row_per_period_and_shares_sum_to_one(self, labelled_frame):
        summary = period_summary(labelled_frame)
        assert len(summary) == labelled_frame[constants.PERIOD_COLUMN].nunique()
        assert summary["share_of_rows"].sum() == pytest.approx(1.0)

    def test_reports_the_base_rate_per_period(self, labelled_frame):
        summary = period_summary(labelled_frame)
        assert summary["subscription_rate"].between(0.0, 1.0).all()

    def test_requires_the_period_column(self, raw_frame):
        with pytest.raises(CalendarReconstructionError, match="missing"):
            period_summary(raw_frame)


class TestMacroPeriodCollinearity:
    def test_macro_features_are_fully_determined_by_the_period(self, labelled_frame):
        """The project's central claim, asserted as a test.

        The fixture builds macro values that are constant within a period, exactly
        as the real dataset does. If a future change to the feature list broke
        that relationship, this test would catch it before the documentation
        started making a claim the data no longer supports.
        """
        collinearity = macro_period_collinearity(labelled_frame)
        assert set(collinearity["feature"]) == set(constants.MACRO_FEATURES)
        assert (collinearity["between_period_variance_share"] > 0.99).all()

    def test_reports_the_number_of_distinct_periods(self, labelled_frame):
        collinearity = macro_period_collinearity(labelled_frame)
        expected = labelled_frame[constants.PERIOD_COLUMN].nunique()
        assert (collinearity["distinct_periods"] == expected).all()


@pytest.mark.requires_dataset
class TestRealDataset:
    def test_reconstructs_the_documented_campaign_window(self, require_dataset):
        """The real file must reconstruct to May 2008 - November 2010, 26 months.

        This is the published extent of the UCI campaign. Pinning it means a
        changed upstream file, or a changed row order, fails loudly instead of
        quietly shifting every temporal result.
        """
        from term_deposit.data.loader import load_raw_dataset

        frame = add_contact_period(load_raw_dataset(require_dataset))
        periods = frame[constants.PERIOD_COLUMN]
        assert str(periods.min()) == "2008-05"
        assert str(periods.max()) == "2010-11"
        assert periods.nunique() == 26
