"""Reconstruct the contact calendar the raw file leaves implicit.

The UCI extract records ``month`` but not the year, which is why almost every
published analysis of this dataset evaluates on a random split. The rows are
ordered by contact date, so the year can be recovered exactly: each time the
month number decreases relative to the previous row, the calendar has rolled over.

This is the enabling step for everything else in the project. Without a calendar
key there is no out-of-time split, no rolling backtest, and no way to see that the
macro indicators are standing in for "which month is this" (ADR 0003).

The reconstruction is a *label* used for splitting and reporting only. It is never
fed to a model — knowing the absolute date of a future call would be its own
source of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from term_deposit import constants
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

#: Upper bound on the number of monthly periods a single campaign can plausibly
#: span. Used only as a sanity check on row ordering, not as a modelling limit.
MAX_PLAUSIBLE_PERIODS = 120


class CalendarReconstructionError(ValueError):
    """Raised when the row order is not consistent with a chronological campaign."""


def reconstruct_contact_period(
    months: pd.Series,
    *,
    start_year: int = constants.CAMPAIGN_START_YEAR,
    strict: bool = True,
) -> pd.PeriodIndex:
    """Derive a monthly period for each row from the campaign's row order.

    Args:
        months: Month abbreviations (``"may"``, ``"jun"``, ...) in contact order.
        start_year: Calendar year of the first row.
        strict: Sanity-check that the reconstructed span is plausible. Guards
            against rows arriving out of contact order, which would silently
            invalidate every temporal claim downstream.

    Returns:
        A monthly :class:`pandas.PeriodIndex` aligned with ``months``.

    Raises:
        CalendarReconstructionError: On unknown month labels, an empty input, or
            (when ``strict``) an implausibly long reconstructed span.

    Example:
        >>> months = pd.Series(["nov", "dec", "mar"])
        >>> [str(p) for p in reconstruct_contact_period(months, start_year=2008)]
        ['2008-11', '2008-12', '2009-03']
    """
    if months.empty:
        msg = "cannot reconstruct a calendar from an empty column"
        raise CalendarReconstructionError(msg)

    numbers = months.map(constants.MONTH_TO_NUMBER)
    if numbers.isna().any():
        unknown = sorted(set(months[numbers.isna()].astype(str).unique()))
        msg = f"unrecognised month label(s): {unknown}"
        raise CalendarReconstructionError(msg)

    month_numbers = numbers.to_numpy(dtype=np.int64)
    # A decrease in month number means the year rolled over.
    rollovers = np.concatenate([[0], (np.diff(month_numbers) < 0).astype(np.int64)])
    years = start_year + np.cumsum(rollovers)

    periods = pd.PeriodIndex(
        [
            pd.Period(year=int(year), month=int(month), freq="M")
            for year, month in zip(years, month_numbers, strict=True)
        ],
        freq="M",
        name=constants.PERIOD_COLUMN,
    )

    if strict:
        _assert_plausible_span(periods)

    logger.debug(
        "Reconstructed %d contact periods spanning %s..%s",
        periods.nunique(),
        periods.min(),
        periods.max(),
    )
    return periods


def _assert_plausible_span(periods: pd.PeriodIndex) -> None:
    """Fail if the reconstructed span is too long to be a real campaign.

    The rollover rule cannot produce a duplicate ``(year, month)`` pair, so the
    thing worth checking is not uniqueness but *plausibility*. The failure this
    guards against is rows arriving out of contact order — a re-sorted export, a
    shuffled sample. Every adjacent month decrease then counts as a year, so a
    shuffled 41k-row file reconstructs to thousands of "years" instead of three.

    Passing this check does not prove the rows are ordered; it rules out the
    catastrophic case loudly rather than letting the out-of-time split silently
    mix future rows into training.
    """
    span = int(periods.nunique())
    if span > MAX_PLAUSIBLE_PERIODS:
        msg = (
            f"reconstructed {span} calendar periods, more than the {MAX_PLAUSIBLE_PERIODS} "
            "a single campaign plausibly spans. The rows are almost certainly not in "
            "contact order, which would make the out-of-time split invalid. "
            "Re-export the data in its original order, or pass strict=False if you "
            "genuinely intend to reconstruct a longer campaign."
        )
        raise CalendarReconstructionError(msg)


def add_contact_period(
    frame: pd.DataFrame,
    *,
    month_column: str = "month",
    start_year: int = constants.CAMPAIGN_START_YEAR,
    strict: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with the reconstructed period column added.

    Args:
        frame: Raw data in contact order.
        month_column: Column holding month abbreviations.
        start_year: Calendar year of the first row.
        strict: Passed through to :func:`reconstruct_contact_period`.

    Returns:
        A copy carrying :data:`~term_deposit.constants.PERIOD_COLUMN`.
    """
    if month_column not in frame.columns:
        msg = f"column {month_column!r} not present; cannot reconstruct the calendar"
        raise CalendarReconstructionError(msg)

    result = frame.copy()
    result[constants.PERIOD_COLUMN] = reconstruct_contact_period(
        frame[month_column], start_year=start_year, strict=strict
    )
    return result


def period_summary(
    frame: pd.DataFrame,
    *,
    label_column: str = constants.LABEL_COLUMN,
    period_column: str = constants.PERIOD_COLUMN,
) -> pd.DataFrame:
    """Per-period volume, base rate and mean macro values.

    This table is the evidence behind the project's central claim: the base rate
    and the macro indicators move together across periods, so a model given both
    can recover the base rate from the macro block alone.
    """
    if period_column not in frame.columns:
        msg = f"{period_column!r} missing; call add_contact_period first"
        raise CalendarReconstructionError(msg)

    aggregations: dict[str, tuple[str, str]] = {"n_contacts": (period_column, "size")}
    if label_column in frame.columns:
        aggregations["subscription_rate"] = (label_column, "mean")
    for macro in constants.MACRO_FEATURES:
        if macro in frame.columns:
            aggregations[macro] = (macro, "mean")

    summary = frame.groupby(period_column, observed=True).agg(**aggregations)
    summary = summary.sort_index()
    summary["share_of_rows"] = summary["n_contacts"] / summary["n_contacts"].sum()
    return summary.reset_index()


def macro_period_collinearity(
    frame: pd.DataFrame,
    *,
    period_column: str = constants.PERIOD_COLUMN,
) -> pd.DataFrame:
    """Quantify how completely each macro feature is determined by the period.

    Returns one row per macro feature with the share of its total variance that
    lies *between* periods. A value near 1.0 means the feature is effectively a
    period identifier and carries no information for ranking customers contacted
    in the same month.
    """
    records = []
    for macro in constants.MACRO_FEATURES:
        if macro not in frame.columns:
            continue
        series = frame[macro]
        total_variance = float(series.var(ddof=0))
        group_means = series.groupby(frame[period_column], observed=True).transform("mean")
        within_variance = float((series - group_means).var(ddof=0))
        between_share = (
            1.0 - within_variance / total_variance if total_variance > 0 else float("nan")
        )
        records.append(
            {
                "feature": macro,
                "between_period_variance_share": between_share,
                "distinct_values": int(series.nunique()),
                "distinct_periods": int(frame[period_column].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)
