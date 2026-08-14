"""Data ingestion, validation and splitting."""

from __future__ import annotations

from term_deposit.data.loader import DatasetNotFoundError, download_raw_dataset, load_raw_dataset
from term_deposit.data.schema import (
    RAW_SCHEMA,
    ColumnSpec,
    SchemaValidationError,
    TableSchema,
    validate_raw_dataframe,
)
from term_deposit.data.splits import DataSplit, SplitIndices, make_split

__all__ = [
    "RAW_SCHEMA",
    "ColumnSpec",
    "DataSplit",
    "DatasetNotFoundError",
    "SchemaValidationError",
    "SplitIndices",
    "TableSchema",
    "download_raw_dataset",
    "load_raw_dataset",
    "make_split",
    "validate_raw_dataframe",
]
