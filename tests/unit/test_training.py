"""Training, calibration, cross-validation and backtesting.

The backtest is what model selection actually rests on, so it needs the same
scrutiny as the metrics do: the folds must be genuinely out-of-time, the
estimator must be refitted per fold, and a degenerate month must not abort a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from term_deposit import constants
from term_deposit.config import (
    AppConfig,
    CalibrationConfig,
    EvaluationConfig,
    ModelSpec,
    PathsConfig,
    SplitConfig,
    TrackingConfig,
    TrainingConfig,
)
from term_deposit.data.splits import make_split
from term_deposit.training.trainer import (
    BacktestResult,
    CrossValidationResult,
    FrozenPipelineCalibrator,
    cross_validate_model,
    rolling_origin_backtest,
    train_all,
    train_model,
)

LOGISTIC = ModelSpec(
    name="logistic_regression",
    estimator="logistic_regression",
    params={"max_iter": 200},
    balance_strategy="class_weight",
)
WEIGHTED_TREES = ModelSpec(
    name="xgboost",
    estimator="xgboost",
    params={"n_estimators": 20, "max_depth": 3},
    balance_strategy="scale_pos_weight",
)


def _config(paths: PathsConfig, **training_overrides: Any) -> AppConfig:
    defaults = {
        "models": (LOGISTIC,),
        "calibration": CalibrationConfig(enabled=False),
        "cross_validate": False,
        "backtest": False,
        "cv_folds": 3,
        "backtest_periods": 4,
        "backtest_min_rows": 50,
        "n_jobs": 1,
    }
    training = TrainingConfig(**{**defaults, **training_overrides})
    return AppConfig(
        paths=paths,
        split=SplitConfig(strategy="out_of_time", test_periods=3, validation_periods=3),
        training=training,
        evaluation=EvaluationConfig(n_bootstrap=0, within_period_min_rows=20),
        tracking=TrackingConfig(backend="none"),
        log_level="WARNING",
    )


@pytest.fixture
def split(labelled_frame, paths):
    config = _config(paths)
    return make_split(labelled_frame, config.split, feature_columns=config.features.input_columns())


class TestTrainModel:
    def test_fits_only_on_the_training_split(self, paths, split):
        model = train_model(LOGISTIC, split, _config(paths))
        assert model.pipeline.named_steps["encode"].named_transformers_[
            "numeric"
        ].n_samples_seen_ == len(split.X_train)

    def test_records_the_fit_duration(self, paths, split):
        assert train_model(LOGISTIC, split, _config(paths)).fit_seconds > 0

    def test_scale_pos_weight_comes_from_the_training_split(self, paths, split):
        model = train_model(WEIGHTED_TREES, split, _config(paths))
        assert model.extras["scale_pos_weight"] == pytest.approx(split.scale_pos_weight())

    def test_unweighted_models_get_no_ratio(self, paths, split):
        assert train_model(LOGISTIC, split, _config(paths)).extras["scale_pos_weight"] is None


class TestCalibration:
    def test_calibration_never_inverts_a_pair(self, paths, split):
        """Calibration fixes the probability scale without disturbing the ranking.

        Isotonic regression is monotone *non-decreasing*, so it may merge two
        scores into one step but must never put a lower-scored customer above a
        higher-scored one. A rank correlation would understate this, because the
        step function deliberately creates ties; the exact property is that no
        strictly-ordered pair changes direction.
        """
        config = _config(paths, calibration=CalibrationConfig(enabled=True, method="isotonic"))
        model = train_model(LOGISTIC, split, config)
        assert model.calibrator is not None

        raw = model.pipeline.predict_proba(split.X_test)[:, 1]
        calibrated = model.predict_proba(split.X_test)

        order = np.argsort(raw, kind="stable")
        ordered_calibrated = calibrated[order]
        assert np.all(np.diff(ordered_calibrated) >= -1e-12), (
            "isotonic calibration reordered customers; the ranking must be preserved"
        )

        # And it does change the scale, otherwise it would be doing nothing.
        assert not np.allclose(raw, calibrated)

    def test_calibrated_probabilities_stay_in_range(self, paths, split):
        calibrator = FrozenPipelineCalibrator(
            train_model(LOGISTIC, split, _config(paths)).pipeline, method="isotonic"
        ).fit(split.X_validation, split.y_validation)
        probabilities = calibrator.predict_proba(split.X_test)
        assert probabilities.shape == (len(split.X_test), 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_sigmoid_calibration_also_works(self, paths, split):
        calibrator = FrozenPipelineCalibrator(
            train_model(LOGISTIC, split, _config(paths)).pipeline, method="sigmoid"
        ).fit(split.X_validation, split.y_validation)
        assert calibrator.predict_proba(split.X_test)[:, 1].max() <= 1.0

    def test_an_unfitted_calibrator_refuses_to_score(self, paths, split):
        calibrator = FrozenPipelineCalibrator(train_model(LOGISTIC, split, _config(paths)).pipeline)
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator.predict_proba(split.X_test)

    def test_calibration_is_skipped_when_validation_is_unusable(self, paths, split):
        """A single-class validation split cannot calibrate anything.

        The run should continue uncalibrated rather than fail, and the artifact's
        notes then record that its probabilities are raw scores.
        """
        degenerate = split.__class__(
            strategy=split.strategy,
            X_train=split.X_train,
            y_train=split.y_train,
            X_validation=split.X_validation,
            y_validation=split.y_validation * 0,
            X_test=split.X_test,
            y_test=split.y_test,
            periods=split.periods,
            boundaries=split.boundaries,
        )
        config = _config(paths, calibration=CalibrationConfig(enabled=True))
        assert train_model(LOGISTIC, degenerate, config).calibrator is None


class TestCrossValidation:
    def test_returns_one_score_per_fold(self, paths, split):
        result = cross_validate_model(LOGISTIC, split, _config(paths))
        assert isinstance(result, CrossValidationResult)
        assert len(result.scores) == 3
        assert result.mean == pytest.approx(float(np.mean(result.scores)))

    def test_scores_the_configured_metric(self, paths, split):
        result = cross_validate_model(LOGISTIC, split, _config(paths), scoring="roc_auc")
        assert result.metric == "roc_auc"
        assert all(0.0 <= score <= 1.0 for score in result.scores)

    def test_is_reproducible(self, paths, split):
        first = cross_validate_model(LOGISTIC, split, _config(paths))
        second = cross_validate_model(LOGISTIC, split, _config(paths))
        assert first.scores == second.scores

    def test_serialises_to_a_plain_dict(self, paths, split):
        payload = cross_validate_model(LOGISTIC, split, _config(paths)).to_dict()
        assert set(payload) == {"metric", "scores", "mean", "std"}


class TestRollingOriginBacktest:
    @pytest.fixture
    def backtest(self, labelled_frame, paths) -> BacktestResult:
        config = _config(paths)
        return rolling_origin_backtest(
            LOGISTIC,
            labelled_frame[list(config.features.input_columns())],
            labelled_frame[constants.LABEL_COLUMN],
            labelled_frame[constants.PERIOD_COLUMN],
            config,
            n_folds=4,
            min_test_rows=50,
        )

    def test_produces_one_record_per_scored_period(self, backtest):
        assert 0 < len(backtest.folds) <= 4

    def test_each_fold_reports_its_period_and_sizes(self, backtest):
        for fold in backtest.folds:
            assert {
                "period",
                "n_train",
                "n_test",
                "base_rate",
                "roc_auc",
                "average_precision",
                "lift_at_20pct",
            } <= set(fold)

    def test_the_training_window_expands_over_folds(self, backtest):
        sizes = [fold["n_train"] for fold in backtest.folds]
        assert sizes == sorted(sizes)

    def test_every_fold_trains_on_more_rows_than_it_scores(self, backtest):
        """Every fold trains on more rows than it scores.

        A fold that trained on less than it tested would mean the expanding
        window had been built backwards.
        """
        assert all(fold["n_train"] > fold["n_test"] for fold in backtest.folds)

    def test_summary_reports_mean_and_spread(self, backtest):
        summary = backtest.summary()
        assert summary["n_folds"] == len(backtest.folds)
        assert "average_precision_mean" in summary
        assert "average_precision_std" in summary

    def test_summary_of_an_empty_backtest_is_empty(self):
        assert BacktestResult(folds=()).summary() == {}

    def test_converts_to_a_frame(self, backtest):
        frame = backtest.to_frame()
        assert len(frame) == len(backtest.folds)
        assert "period" in frame.columns

    def test_skips_periods_with_too_few_rows(self, labelled_frame, paths):
        config = _config(paths)
        result = rolling_origin_backtest(
            LOGISTIC,
            labelled_frame[list(config.features.input_columns())],
            labelled_frame[constants.LABEL_COLUMN],
            labelled_frame[constants.PERIOD_COLUMN],
            config,
            n_folds=4,
            min_test_rows=10_000,
        )
        assert result.folds == ()


class TestTrainAll:
    def test_returns_one_model_per_enabled_spec(self, paths, split):
        config = _config(paths)
        models = train_all(split, config)
        assert [m.name for m in models] == [
            s.name for s in config.require_training().enabled_models
        ]

    def test_attaches_cross_validation_when_enabled(self, paths, split):
        models = train_all(split, _config(paths, cross_validate=True))
        assert models[0].cross_validation is not None

    def test_attaches_the_backtest_when_inputs_are_supplied(self, labelled_frame, paths, split):
        config = _config(paths, backtest=True)
        models = train_all(
            split,
            config,
            frame=labelled_frame[list(config.features.input_columns())],
            labels=labelled_frame[constants.LABEL_COLUMN],
            periods=labelled_frame[constants.PERIOD_COLUMN],
        )
        assert models[0].backtest is not None
        assert models[0].backtest.folds

    def test_skips_the_backtest_when_inputs_are_missing(self, paths, split):
        """Requested but unsatisfiable: warn and continue rather than crash a run."""
        models = train_all(split, _config(paths, backtest=True))
        assert models[0].backtest is None
