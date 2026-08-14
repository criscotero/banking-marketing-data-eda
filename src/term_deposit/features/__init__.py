"""Calendar reconstruction and the preprocessing pipeline."""

from __future__ import annotations

from term_deposit.features.calendar import (
    CalendarReconstructionError,
    add_contact_period,
    period_summary,
    reconstruct_contact_period,
)
from term_deposit.features.pipeline import build_preprocessor, output_feature_names
from term_deposit.features.transformers import PdaysSentinelEncoder

__all__ = [
    "CalendarReconstructionError",
    "PdaysSentinelEncoder",
    "add_contact_period",
    "build_preprocessor",
    "output_feature_names",
    "period_summary",
    "reconstruct_contact_period",
]
