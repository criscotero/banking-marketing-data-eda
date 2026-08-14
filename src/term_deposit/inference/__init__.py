"""Batch and single-record scoring against a saved artifact."""

from __future__ import annotations

from term_deposit.inference.predictor import Predictor, PredictorError, load_predictor
from term_deposit.inference.schema import (
    CustomerRecord,
    ScoredCustomer,
    ScoringRequest,
    ScoringResponse,
)

__all__ = [
    "CustomerRecord",
    "Predictor",
    "PredictorError",
    "ScoredCustomer",
    "ScoringRequest",
    "ScoringResponse",
    "load_predictor",
]
