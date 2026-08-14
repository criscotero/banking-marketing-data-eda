"""Declarative schema for the raw dataset.

The contract is checked at the boundary — once, when the CSV is read — so that
every downstream module can assume well-formed input instead of defending against
it. Validation collects *all* violations before raising, because discovering
problems one run at a time is the slowest way to fix a broken extract.

A hand-rolled schema is used rather than a DataFrame-validation library: the rules
here are few and specific, and keeping them dependency-free means `uv sync`
installs nothing extra to run the checks in CI. See ADR 0007.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from term_deposit import constants


class SchemaValidationError(ValueError):
    """Raised when a DataFrame violates its declared schema.

    Attributes:
        violations: Human-readable descriptions, one per failed rule.
    """

    def __init__(self, violations: Sequence[str]) -> None:
        """Store the violations and render them as a single readable message."""
        self.violations = tuple(violations)
        joined = "\n  - ".join(self.violations)
        super().__init__(f"dataset failed schema validation:\n  - {joined}")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Rules for a single column.

    Attributes:
        name: Column name as it appears in the raw file.
        kind: ``"numeric"`` or ``"categorical"``.
        allowed: Permitted values for categorical columns.
        minimum: Inclusive lower bound for numeric columns.
        maximum: Inclusive upper bound for numeric columns.
        nullable: Whether nulls are tolerated. Every raw column is non-nullable.
    """

    name: str
    kind: str
    allowed: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    nullable: bool = False

    def validate(self, series: pd.Series) -> list[str]:
        """Return a list of violations for ``series`` (empty when it conforms)."""
        problems: list[str] = []

        null_count = int(series.isna().sum())
        if null_count and not self.nullable:
            problems.append(f"{self.name}: {null_count} null value(s) in a non-nullable column")

        non_null = series.dropna()
        if self.kind == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                problems.append(f"{self.name}: expected a numeric dtype, found {series.dtype}")
                return problems
            if self.minimum is not None:
                below = int((non_null < self.minimum).sum())
                if below:
                    problems.append(f"{self.name}: {below} value(s) below minimum {self.minimum}")
            if self.maximum is not None:
                above = int((non_null > self.maximum).sum())
                if above:
                    problems.append(f"{self.name}: {above} value(s) above maximum {self.maximum}")
        elif self.allowed is not None:
            unexpected = sorted(set(non_null.astype(str).unique()) - set(self.allowed))
            if unexpected:
                shown = unexpected[:5]
                suffix = f" (+{len(unexpected) - len(shown)} more)" if len(unexpected) > 5 else ""
                problems.append(f"{self.name}: unexpected categories {shown}{suffix}")

        return problems


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A set of column specs plus table-level expectations."""

    columns: tuple[ColumnSpec, ...]
    expected_rows: int | None = None
    #: Extra columns are tolerated (derived columns are added downstream);
    #: missing columns never are.
    required: tuple[str, ...] = field(default_factory=tuple)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of every declared column."""
        return tuple(spec.name for spec in self.columns)

    def validate(self, frame: pd.DataFrame, *, check_row_count: bool = True) -> list[str]:
        """Return every violation found in ``frame``.

        Args:
            frame: The DataFrame to check.
            check_row_count: Compare against ``expected_rows``. Disabled for
                fixtures and samples, which are deliberately smaller.
        """
        problems: list[str] = []

        required = self.required or self.column_names
        missing = [name for name in required if name not in frame.columns]
        if missing:
            problems.append(f"missing required column(s): {missing}")

        if check_row_count and self.expected_rows is not None and len(frame) != self.expected_rows:
            problems.append(f"expected {self.expected_rows} rows, found {len(frame)}")

        if frame.empty:
            problems.append("dataset is empty")

        for spec in self.columns:
            if spec.name in frame.columns:
                problems.extend(spec.validate(frame[spec.name]))

        return problems


def _numeric(name: str, minimum: float | None = None, maximum: float | None = None) -> ColumnSpec:
    return ColumnSpec(name=name, kind="numeric", minimum=minimum, maximum=maximum)


def _categorical(name: str) -> ColumnSpec:
    return ColumnSpec(name=name, kind="categorical", allowed=constants.CATEGORY_DOMAINS[name])


#: Contract for ``bank-additional-full.csv`` as published by UCI.
#: Bounds are deliberately generous — they catch corruption and unit changes
#: (an age of 900, a negative call count), not legitimate distribution shift.
RAW_SCHEMA = TableSchema(
    columns=(
        _numeric("age", minimum=17, maximum=120),
        _categorical("job"),
        _categorical("marital"),
        _categorical("education"),
        _categorical("default"),
        _categorical("housing"),
        _categorical("loan"),
        _categorical("contact"),
        _categorical("month"),
        _categorical("day_of_week"),
        _numeric("duration", minimum=0),
        _numeric("campaign", minimum=1),
        _numeric("pdays", minimum=0, maximum=constants.PDAYS_NEVER_CONTACTED),
        _numeric("previous", minimum=0),
        _categorical("poutcome"),
        _numeric("emp.var.rate"),
        _numeric("cons.price.idx", minimum=0),
        _numeric("cons.conf.idx"),
        _numeric("euribor3m", minimum=0),
        _numeric("nr.employed", minimum=0),
        ColumnSpec(
            name=constants.TARGET_COLUMN,
            kind="categorical",
            allowed=(constants.TARGET_NEGATIVE_LABEL, constants.TARGET_POSITIVE_LABEL),
        ),
    ),
    expected_rows=constants.RAW_ROW_COUNT,
)


def validate_raw_dataframe(
    frame: pd.DataFrame,
    *,
    schema: TableSchema = RAW_SCHEMA,
    check_row_count: bool = True,
) -> pd.DataFrame:
    """Validate a raw DataFrame and return it unchanged.

    Args:
        frame: Freshly parsed raw data.
        schema: Contract to enforce. Defaults to :data:`RAW_SCHEMA`.
        check_row_count: Whether the exact row count is part of the contract.

    Returns:
        The same object, so the call can be chained onto a read.

    Raises:
        SchemaValidationError: If any rule fails. All failures are reported at once.
    """
    problems = schema.validate(frame, check_row_count=check_row_count)
    if problems:
        raise SchemaValidationError(problems)
    return frame


def summarise_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column data-quality summary used by the EDA notebook and reports.

    Reports the literal ``"unknown"`` category separately from true nulls, because
    in this dataset they mean different things: ``"unknown"`` is a recorded
    non-answer, and the project keeps it as a signal rather than imputing it.
    """
    records = []
    for column in frame.columns:
        series = frame[column]
        unknown = (
            int((series.astype(str) == constants.UNKNOWN_CATEGORY).sum())
            if not pd.api.types.is_numeric_dtype(series)
            else 0
        )
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "pct_missing": float(series.isna().mean() * 100),
                "n_unknown": unknown,
                "pct_unknown": float(unknown / len(frame) * 100) if len(frame) else np.nan,
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(records)
