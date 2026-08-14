"""Fitting models under a protocol that keeps evaluation honest.

Three rules hold everywhere in this module:

1. **Nothing is fitted on the test split** — not the scaler, not the encoder, not
   the calibrator, not the threshold. Preprocessing lives inside the pipeline, so
   this is enforced by construction rather than by discipline.
2. **Quantities derived from labels come from the training split only.**
   ``scale_pos_weight`` is computed from training rows, not from the full dataset.
3. **Every fold gets a fresh estimator.** ``clone`` is used before each fit, so no
   fitted state leaks between folds or between models.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from term_deposit.config import AppConfig, ModelSpec
from term_deposit.data.splits import DataSplit, rolling_origin_folds
from term_deposit.evaluation.metrics import binary_metrics
from term_deposit.models.registry import build_pipeline
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)


class FrozenPipelineCalibrator:
    """Calibrate a fitted pipeline on held-out data without refitting it.

    ``CalibratedClassifierCV(cv="prefit")`` would refit the whole pipeline's
    preprocessing against the calibration split. Here the fitted pipeline is
    treated as a fixed scorer and only a univariate mapping from its output to a
    probability is learned, so the training-time transformation is preserved
    exactly and the calibration split contributes nothing but its labels.

    Class-weighted estimators need this: weighting shifts the probability scale
    upwards, which is harmless for ranking but makes every reported probability —
    and therefore every expected-value calculation — wrong.
    """

    def __init__(self, pipeline: Pipeline, method: str = "isotonic") -> None:
        """Wrap a fitted pipeline with the named calibration method."""
        self.pipeline = pipeline
        self.method = method
        self._calibrator: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FrozenPipelineCalibrator:  # noqa: N803
        """Learn the score-to-probability mapping on a held-out split."""
        raw_scores = self.pipeline.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw_scores, np.asarray(y, dtype=float))
        else:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(C=1e10, solver="lbfgs")
            calibrator.fit(raw_scores.reshape(-1, 1), np.asarray(y))
        self._calibrator = calibrator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Calibrated positive-class probabilities, as a two-column matrix."""
        if self._calibrator is None:
            msg = "calibrator has not been fitted"
            raise RuntimeError(msg)
        raw_scores = self.pipeline.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            positive = np.asarray(self._calibrator.predict(raw_scores), dtype=float)
        else:
            positive = np.asarray(
                self._calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=float
            )
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Hard predictions at 0.5.

        Present for scikit-learn duck-typing only; the pipeline always scores
        with the threshold selected on validation, never with this default.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Cross-validated scores on the training split."""

    metric: str
    scores: tuple[float, ...]
    mean: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view."""
        return {
            "metric": self.metric,
            "scores": list(self.scores),
            "mean": self.mean,
            "std": self.std,
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Rolling-origin backtest: one fold per trailing calendar period.

    Averaging across folds is a far more stable estimate than a single held-out
    window, and the per-fold spread shows how much performance depends on which
    month happens to be held out.
    """

    folds: tuple[dict[str, Any], ...]

    def to_frame(self) -> pd.DataFrame:
        """Per-fold results as a DataFrame."""
        return pd.DataFrame(list(self.folds))

    def summary(self) -> dict[str, float]:
        """Mean and standard deviation of each fold metric."""
        frame = self.to_frame()
        if frame.empty:
            return {}
        result: dict[str, float] = {"n_folds": float(len(frame))}
        for column in ("roc_auc", "average_precision", "lift_at_20pct", "base_rate"):
            if column in frame.columns:
                result[f"{column}_mean"] = float(frame[column].mean())
                result[f"{column}_std"] = float(frame[column].std(ddof=0))
        return result


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted model plus everything learned about it during training."""

    name: str
    spec: ModelSpec
    pipeline: Pipeline
    calibrator: FrozenPipelineCalibrator | None
    fit_seconds: float
    cross_validation: CrossValidationResult | None = None
    backtest: BacktestResult | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def scorer(self) -> Any:
        """The object used to produce probabilities: calibrated when available."""
        return self.calibrator if self.calibrator is not None else self.pipeline

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Positive-class probabilities from the scorer."""
        return np.asarray(self.scorer.predict_proba(X))[:, 1]


def train_model(
    spec: ModelSpec,
    split: DataSplit,
    config: AppConfig,
    *,
    calibrate: bool | None = None,
) -> TrainedModel:
    """Fit one model on the training split and calibrate it on validation.

    Args:
        spec: Model definition.
        split: Materialised split. Only ``X_train``/``y_train`` are fitted on;
            ``X_validation``/``y_validation`` are used for calibration only.
        config: Resolved application configuration.
        calibrate: Override the configured calibration flag. Calibration is
            skipped automatically when the validation split is empty or
            single-class.

    Returns:
        A :class:`TrainedModel`.
    """
    training = config.require_training()
    scale_pos_weight = (
        split.scale_pos_weight() if spec.balance_strategy == "scale_pos_weight" else None
    )

    pipeline = build_pipeline(
        spec,
        config.features,
        random_state=config.random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=training.n_jobs,
    )

    started = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - started
    logger.info("Fitted %s in %.1fs on %d rows", spec.name, fit_seconds, len(split.X_train))

    should_calibrate = training.calibration.enabled if calibrate is None else calibrate
    calibrator: FrozenPipelineCalibrator | None = None
    if should_calibrate:
        if len(split.y_validation) == 0 or split.y_validation.nunique() < 2:
            logger.warning(
                "Skipping calibration for %s: the validation split is empty or single-class",
                spec.name,
            )
        else:
            calibrator = FrozenPipelineCalibrator(pipeline, method=training.calibration.method).fit(
                split.X_validation, split.y_validation
            )
            logger.debug("Calibrated %s with %s regression", spec.name, training.calibration.method)

    return TrainedModel(
        name=spec.name,
        spec=spec,
        pipeline=pipeline,
        calibrator=calibrator,
        fit_seconds=fit_seconds,
        extras={"scale_pos_weight": scale_pos_weight},
    )


def cross_validate_model(
    spec: ModelSpec,
    split: DataSplit,
    config: AppConfig,
    *,
    scoring: str = "average_precision",
) -> CrossValidationResult:
    """Stratified k-fold cross-validation on the training split.

    Uses an unfitted clone per fold, so the preprocessing is refitted inside each
    fold and cannot see the fold's held-out rows.

    Note:
        Under the out-of-time protocol these folds are still *random* within the
        training window, so they measure in-sample stability rather than temporal
        generalisation. :func:`rolling_origin_backtest` is the temporal answer;
        both are reported because they say different things.
    """
    training = config.require_training()
    scale_pos_weight = (
        split.scale_pos_weight() if spec.balance_strategy == "scale_pos_weight" else None
    )
    pipeline = build_pipeline(
        spec,
        config.features,
        random_state=config.random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,  # parallelism belongs to the outer cross_val_score loop
    )
    folds = StratifiedKFold(
        n_splits=training.cv_folds, shuffle=True, random_state=config.random_state
    )
    scores = cross_val_score(
        clone(pipeline),
        split.X_train,
        split.y_train,
        cv=folds,
        scoring=scoring,
        n_jobs=training.n_jobs,
    )
    result = CrossValidationResult(
        metric=scoring,
        scores=tuple(float(value) for value in scores),
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
    )
    logger.info("CV %s for %s: %.4f +/- %.4f", scoring, spec.name, result.mean, result.std)
    return result


def rolling_origin_backtest(
    spec: ModelSpec,
    frame: pd.DataFrame,
    labels: pd.Series,
    periods: pd.Series,
    config: AppConfig,
    *,
    n_folds: int,
    min_test_rows: int = 100,
) -> BacktestResult:
    """Retrain on all prior periods and score the next one, fold by fold.

    This is the closest offline simulation of monthly retraining in production,
    and it is where the difference between the random and out-of-time protocols
    shows up as a spread of real numbers rather than a single point estimate.

    Args:
        spec: Model definition.
        frame: Feature frame for the whole dataset.
        labels: Binary labels aligned with ``frame``.
        periods: Calendar period per row.
        config: Resolved application configuration.
        n_folds: Number of trailing periods to score.
        min_test_rows: Skip periods smaller than this.

    Returns:
        A :class:`BacktestResult` with one record per scored period.
    """
    training = config.require_training()
    folds = rolling_origin_folds(periods, n_folds=n_folds, min_test_rows=min_test_rows)
    records: list[dict[str, Any]] = []

    for train_index, test_index, period in folds:
        y_train = labels.loc[train_index]
        y_test = labels.loc[test_index]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            logger.debug("Skipping backtest fold %s: a split is single-class", period)
            continue

        positive_rate = float(y_train.mean())
        scale_pos_weight = (
            (1.0 - positive_rate) / positive_rate
            if spec.balance_strategy == "scale_pos_weight" and 0.0 < positive_rate < 1.0
            else None
        )
        pipeline = build_pipeline(
            spec,
            config.features,
            random_state=config.random_state,
            scale_pos_weight=scale_pos_weight,
            n_jobs=training.n_jobs,
        )
        pipeline.fit(frame.loc[train_index], y_train)
        scores = pipeline.predict_proba(frame.loc[test_index])[:, 1]
        metrics = binary_metrics(y_test, scores, top_k_fractions=config.evaluation.top_k_fractions)
        records.append(
            {
                "period": period,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "base_rate": metrics.base_rate,
                "roc_auc": metrics.roc_auc,
                "average_precision": metrics.average_precision,
                "lift_at_20pct": metrics.top_k.get("0.20", {}).get("lift", float("nan")),
                "brier_score": metrics.brier_score,
            }
        )
        logger.debug(
            "Backtest %s @ %s: AP=%.4f lift@20%%=%.2f",
            spec.name,
            period,
            metrics.average_precision,
            records[-1]["lift_at_20pct"],
        )

    logger.info("Backtested %s over %d folds", spec.name, len(records))
    return BacktestResult(folds=tuple(records))


def train_all(
    split: DataSplit,
    config: AppConfig,
    *,
    frame: pd.DataFrame | None = None,
    labels: pd.Series | None = None,
    periods: pd.Series | None = None,
    specs: Sequence[ModelSpec] | None = None,
) -> list[TrainedModel]:
    """Train every enabled model under one protocol.

    Args:
        split: Materialised split shared by all models, so the comparison is fair.
        config: Resolved application configuration.
        frame: Full feature frame, required for the backtest stage.
        labels: Full label series, required for the backtest stage.
        periods: Full period series, required for the backtest stage.
        specs: Override the configured model list.

    Returns:
        One :class:`TrainedModel` per enabled spec, in configuration order.
    """
    training = config.require_training()
    selected = tuple(specs) if specs is not None else training.enabled_models
    backtest_inputs_present = frame is not None and labels is not None and periods is not None

    trained: list[TrainedModel] = []
    for spec in selected:
        model = train_model(spec, split, config)

        cross_validation = (
            cross_validate_model(spec, split, config, scoring=config.evaluation.primary_metric)
            if training.cross_validate
            else None
        )

        backtest: BacktestResult | None = None
        if training.backtest:
            if backtest_inputs_present:
                backtest = rolling_origin_backtest(
                    spec,
                    frame,  # type: ignore[arg-type]
                    labels,  # type: ignore[arg-type]
                    periods,  # type: ignore[arg-type]
                    config,
                    n_folds=training.backtest_periods,
                    min_test_rows=training.backtest_min_rows,
                )
            else:
                logger.warning(
                    "training.backtest is enabled but the full frame/labels/periods "
                    "were not supplied; skipping the backtest"
                )

        trained.append(
            TrainedModel(
                name=model.name,
                spec=model.spec,
                pipeline=model.pipeline,
                calibrator=model.calibrator,
                fit_seconds=model.fit_seconds,
                cross_validation=cross_validation,
                backtest=backtest,
                extras=model.extras,
            )
        )
    return trained
