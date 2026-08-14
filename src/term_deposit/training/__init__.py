"""Model fitting, calibration, cross-validation and backtesting."""

from __future__ import annotations

from term_deposit.training.trainer import (
    BacktestResult,
    CrossValidationResult,
    TrainedModel,
    cross_validate_model,
    rolling_origin_backtest,
    train_all,
    train_model,
)

__all__ = [
    "BacktestResult",
    "CrossValidationResult",
    "TrainedModel",
    "cross_validate_model",
    "rolling_origin_backtest",
    "train_all",
    "train_model",
]
