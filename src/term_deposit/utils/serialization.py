"""JSON serialization for metric and metadata documents.

Metric payloads are full of numpy scalars, pandas Periods and Paths, none of
which `json` handles. `to_jsonable` normalises them once so that every report
written by the package is plain, diffable JSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas/pathlib values into JSON-native equivalents.

    Non-finite floats become ``None`` rather than the ``NaN`` literal, which is
    valid Python but invalid JSON and breaks strict parsers downstream.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(record) for record in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence | set | frozenset):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):  # pydantic models
        return to_jsonable(value.model_dump())
    return str(value)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories.

    Keys are left in insertion order (not sorted) so that generated reports read
    in the order the pipeline produced them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), indent=indent, ensure_ascii=False)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> Path:
    """Append one JSON object as a line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_jsonable(payload), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
    return path
