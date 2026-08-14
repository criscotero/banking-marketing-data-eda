"""Train/validation/test strategies.

Two protocols are implemented and both are reported, because the difference
between them *is* the finding (ADR 0002):

``random``
    Stratified shuffle, as in the original notebook and in most published
    analyses of this dataset. Training and test rows are drawn from the same
    calendar months, so the model can learn each month's base rate from the macro
    indicators and replay it on the test set.

``out_of_time``
    Train on the earliest months, validate on the next block, test on the most
    recent months. This is the only protocol that answers the question the bank
    actually has: *will a model fitted on history rank the customers we call next
    quarter?*

Both return the same object, so downstream code never branches on the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from term_deposit import constants
from term_deposit.config import SplitConfig
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

SplitName = Literal["train", "validation", "test"]


class SplitError(ValueError):
    """Raised when a split cannot be produced as configured."""


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Row labels assigned to each split."""

    train: pd.Index
    validation: pd.Index
    test: pd.Index

    def __post_init__(self) -> None:
        """Reject overlapping partitions, which would contaminate evaluation."""
        overlaps = {
            "train/validation": self.train.intersection(self.validation),
            "train/test": self.train.intersection(self.test),
            "validation/test": self.validation.intersection(self.test),
        }
        leaking = {name: len(index) for name, index in overlaps.items() if len(index)}
        if leaking:
            msg = f"splits overlap, which would contaminate evaluation: {leaking}"
            raise SplitError(msg)

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per split."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@dataclass(frozen=True, slots=True)
class DataSplit:
    """A materialised split: features, labels and period key per partition.

    Attributes:
        strategy: Which protocol produced it.
        X_train / X_validation / X_test: Feature frames.
        y_train / y_validation / y_test: Binary labels.
        periods: Calendar period per row for the whole dataset, used for
            within-period reporting and for the rolling backtest.
        boundaries: Period cut points, populated for the out-of-time protocol.
    """

    strategy: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    periods: pd.Series
    boundaries: dict[str, str]

    @property
    def positive_rates(self) -> dict[str, float]:
        """Base rate per split.

        A large gap between train and test is the signature of a regime shift,
        and it explains most of the calibration error under the temporal protocol.
        """
        return {
            "train": float(self.y_train.mean()),
            "validation": float(self.y_validation.mean())
            if len(self.y_validation)
            else float("nan"),
            "test": float(self.y_test.mean()),
        }

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per split."""
        return {
            "train": len(self.X_train),
            "validation": len(self.X_validation),
            "test": len(self.X_test),
        }

    def periods_for(self, part: SplitName) -> pd.Series:
        """Calendar periods aligned with one partition's rows."""
        frame = {"train": self.X_train, "validation": self.X_validation, "test": self.X_test}[part]
        return self.periods.loc[frame.index]

    def scale_pos_weight(self) -> float:
        """Negative-to-positive ratio in the training split.

        Computed from training rows only. Deriving it from the full dataset — as
        the original notebook did — lets the class balance of the test set influence
        a training hyperparameter.
        """
        positive_rate = float(self.y_train.mean())
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            msg = f"training split has a degenerate positive rate ({positive_rate})"
            raise SplitError(msg)
        return (1.0 - positive_rate) / positive_rate

    def summary(self) -> dict[str, object]:
        """Serialisable description, recorded with every run's metadata."""
        return {
            "strategy": self.strategy,
            "sizes": self.sizes,
            "positive_rates": self.positive_rates,
            "boundaries": self.boundaries,
            "n_features": self.X_train.shape[1],
        }


def make_split(
    frame: pd.DataFrame,
    config: SplitConfig,
    *,
    feature_columns: tuple[str, ...],
    label_column: str = constants.LABEL_COLUMN,
    period_column: str = constants.PERIOD_COLUMN,
) -> DataSplit:
    """Split ``frame`` according to ``config``.

    Args:
        frame: Full dataset, including the label and the reconstructed period.
        config: Split settings.
        feature_columns: Columns kept as model input.
        label_column: Binary target column.
        period_column: Calendar key produced by the calendar module.

    Returns:
        A :class:`DataSplit`.

    Raises:
        SplitError: If required columns are absent or the configuration cannot be
            satisfied (for example, more held-out periods than the data contains).
    """
    for column in (label_column, period_column):
        if column not in frame.columns:
            msg = f"column {column!r} is required to build a split"
            raise SplitError(msg)

    missing_features = [name for name in feature_columns if name not in frame.columns]
    if missing_features:
        msg = f"feature column(s) not present in the dataset: {missing_features}"
        raise SplitError(msg)

    features = frame.loc[:, list(feature_columns)]
    labels = frame[label_column].astype("int8")
    periods = frame[period_column]

    if config.strategy == "random":
        indices, boundaries = _random_indices(labels, config)
    elif config.strategy == "out_of_time":
        indices, boundaries = _out_of_time_indices(periods, config)
    else:  # pragma: no cover - unreachable while SplitStrategy stays a Literal
        # Kept as a runtime guard: the Literal makes mypy consider this dead,
        # but a new strategy added to the enum without a branch here would
        # otherwise fall through silently.
        msg = f"unknown split strategy {config.strategy!r}"  # type: ignore[unreachable]
        raise SplitError(msg)

    split = DataSplit(
        strategy=config.strategy,
        X_train=features.loc[indices.train],
        y_train=labels.loc[indices.train],
        X_validation=features.loc[indices.validation],
        y_validation=labels.loc[indices.validation],
        X_test=features.loc[indices.test],
        y_test=labels.loc[indices.test],
        periods=periods,
        boundaries=boundaries,
    )
    logger.info(
        "Split=%s sizes=%s positive_rates=%s",
        split.strategy,
        split.sizes,
        {k: round(v, 4) for k, v in split.positive_rates.items()},
    )
    return split


def _random_indices(labels: pd.Series, config: SplitConfig) -> tuple[SplitIndices, dict[str, str]]:
    """Stratified shuffle split into train / validation / test."""
    train_pool, test_index = train_test_split(
        labels.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )
    if config.validation_size <= 0:
        return (
            SplitIndices(train=train_pool, validation=labels.index[:0], test=test_index),
            {"note": "stratified shuffle; no calendar boundary"},
        )

    # Express the validation fraction relative to the remaining pool so that the
    # configured value stays a fraction of the *whole* dataset.
    relative = config.validation_size / (1.0 - config.test_size)
    train_index, validation_index = train_test_split(
        train_pool,
        test_size=relative,
        random_state=config.random_state,
        stratify=labels.loc[train_pool],
    )
    return (
        SplitIndices(train=train_index, validation=validation_index, test=test_index),
        {"note": "stratified shuffle; no calendar boundary"},
    )


def _out_of_time_indices(
    periods: pd.Series, config: SplitConfig
) -> tuple[SplitIndices, dict[str, str]]:
    """Chronological split on calendar-period boundaries."""
    ordered = pd.Index(sorted(periods.unique()))
    required = config.test_periods + config.validation_periods + 1
    if len(ordered) < required:
        msg = (
            f"out-of-time split needs at least {required} calendar periods "
            f"({config.validation_periods} validation + {config.test_periods} test + 1 train), "
            f"but the data covers {len(ordered)}"
        )
        raise SplitError(msg)

    test_periods = set(ordered[-config.test_periods :])
    validation_start = len(ordered) - config.test_periods - config.validation_periods
    validation_periods = set(ordered[validation_start : len(ordered) - config.test_periods])

    is_test = periods.isin(test_periods)
    is_validation = periods.isin(validation_periods)
    is_train = ~(is_test | is_validation)

    boundaries = {
        "train_end": str(ordered[validation_start - 1]),
        "validation_start": str(ordered[validation_start]),
        "validation_end": str(ordered[len(ordered) - config.test_periods - 1]),
        "test_start": str(ordered[-config.test_periods]),
        "test_end": str(ordered[-1]),
    }
    return (
        SplitIndices(
            train=periods.index[is_train],
            validation=periods.index[is_validation],
            test=periods.index[is_test],
        ),
        boundaries,
    )


def rolling_origin_folds(
    periods: pd.Series,
    *,
    n_folds: int,
    min_test_rows: int = 100,
) -> list[tuple[pd.Index, pd.Index, str]]:
    """Build expanding-window backtest folds, one per trailing calendar period.

    Fold *k* trains on every row strictly before period *k* and tests on period *k*.
    This mirrors monthly retraining, and averaging over folds gives a far more
    honest performance estimate than a single arbitrary cut-off.

    Args:
        periods: Calendar period per row.
        n_folds: How many trailing periods to score.
        min_test_rows: Skip periods with fewer rows, where metrics would be noise.

    Returns:
        ``(train_index, test_index, period_label)`` triples, oldest first.
    """
    ordered = sorted(periods.unique())
    folds: list[tuple[pd.Index, pd.Index, str]] = []
    for period in ordered[-n_folds:]:
        test_mask = periods == period
        train_mask = periods < period
        if int(test_mask.sum()) < min_test_rows or not train_mask.any():
            logger.debug("Skipping backtest fold %s (too few rows)", period)
            continue
        folds.append((periods.index[train_mask], periods.index[test_mask], str(period)))
    return folds


def stratified_group_check(labels: pd.Series, periods: pd.Series) -> pd.DataFrame:
    """Base rate per period — the table that motivates the out-of-time protocol."""
    return (
        pd.DataFrame({"period": periods.astype(str), "label": labels})
        .groupby("period", observed=True)["label"]
        .agg(n="size", positive_rate="mean")
        .reset_index()
        .assign(period=lambda d: d["period"].astype(str))
        .sort_values("period")
        .reset_index(drop=True)
    )


def describe_drift(split: DataSplit) -> pd.DataFrame:
    """Compare per-feature means between train and test.

    A large standardised gap on the macro block is the numeric statement of the
    project's central claim: the test window is a different economic regime.
    """
    records = []
    for column in split.X_train.columns:
        train_values = split.X_train[column]
        test_values = split.X_test[column]
        if not pd.api.types.is_numeric_dtype(train_values):
            continue
        train_mean = float(train_values.mean())
        test_mean = float(test_values.mean())
        pooled_std = float(np.sqrt((train_values.var(ddof=0) + test_values.var(ddof=0)) / 2.0))
        records.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "standardised_gap": (
                    abs(test_mean - train_mean) / pooled_std if pooled_std > 0 else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values("standardised_gap", ascending=False)
        .reset_index(drop=True)
    )
