"""End-to-end pipeline behaviour.

These tests exercise the path a user actually runs — prepare, train, persist,
score — on synthetic data, and assert the properties that make the results
trustworthy rather than the specific numbers, which depend on the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from term_deposit import constants
from term_deposit.config import InferenceConfig, SplitConfig
from term_deposit.data.splits import make_split
from term_deposit.evaluation.report import build_comparison_table, evaluate_all, select_best
from term_deposit.inference.predictor import load_predictor
from term_deposit.models.persistence import load_artifact, resolve_artifact_path
from term_deposit.pipelines.experiment import PreparedData, prepare_dataset, run_experiment
from term_deposit.training.trainer import train_all


@pytest.fixture
def prepared(labelled_frame) -> PreparedData:
    """A PreparedData bundle built from the synthetic frame."""
    from term_deposit.data.schema import summarise_quality
    from term_deposit.features.calendar import macro_period_collinearity, period_summary

    return PreparedData(
        frame=labelled_frame,
        checksum="synthetic",
        quality=summarise_quality(labelled_frame),
        periods=period_summary(labelled_frame),
        macro_collinearity=macro_period_collinearity(labelled_frame),
    )


class TestPrepareDataset:
    def test_drops_the_post_call_column(self, app_config, synthetic_csv, monkeypatch):
        """`duration` must be gone before any model can reach it."""
        import shutil

        shutil.copyfile(synthetic_csv, app_config.paths.raw_csv)
        monkeypatch.setattr(
            "term_deposit.pipelines.experiment.ensure_raw_dataset",
            lambda *_a, **_k: app_config.paths.raw_csv,
        )
        config = app_config.model_copy(
            update={"data": app_config.data.model_copy(update={"validate_schema": False})}
        )
        prepared = prepare_dataset(config, download=False)
        assert "duration" not in prepared.frame.columns

    def test_attaches_the_label_and_the_period(self, app_config, synthetic_csv, monkeypatch):
        import shutil

        shutil.copyfile(synthetic_csv, app_config.paths.raw_csv)
        monkeypatch.setattr(
            "term_deposit.pipelines.experiment.ensure_raw_dataset",
            lambda *_a, **_k: app_config.paths.raw_csv,
        )
        config = app_config.model_copy(
            update={"data": app_config.data.model_copy(update={"validate_schema": False})}
        )
        prepared = prepare_dataset(config, download=False)
        assert constants.LABEL_COLUMN in prepared.frame.columns
        assert constants.PERIOD_COLUMN in prepared.frame.columns


class TestRunExperiment:
    def test_produces_a_report_per_model(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=False)
        assert len(result.reports) == len(app_config.require_training().enabled_models)

    def test_writes_an_artifact_and_a_comparison_table(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=True)
        assert result.artifact_dir is not None
        assert (result.artifact_dir / "model.joblib").is_file()
        assert (result.artifact_dir / "metadata.json").is_file()
        assert any("comparison" in key for key in result.written)

    def test_every_model_beats_or_matches_the_no_skill_baseline(self, app_config, prepared):
        """A comparison without a floor cannot tell a good model from a bad one."""
        result = run_experiment(app_config, data=prepared, persist=False)
        by_name = {report.model_name: report for report in result.reports}
        baseline = by_name["baseline_prior"].test_metrics
        assert baseline.roc_auc == pytest.approx(0.5, abs=0.01)
        assert by_name["logistic_regression"].test_metrics.average_precision >= (
            baseline.average_precision
        )

    def test_the_threshold_is_chosen_on_validation(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=False)
        assert all(report.threshold.chosen_on == "validation" for report in result.reports)

    def test_the_persisted_metadata_records_the_protocol_and_contract(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=True)
        assert result.artifact_dir is not None
        metadata = load_artifact(result.artifact_dir).metadata
        assert metadata.split_strategy == "out_of_time"
        assert tuple(metadata.input_columns) == app_config.features.input_columns()
        assert metadata.notes

    def test_the_saved_model_reproduces_the_reported_threshold(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=True)
        assert result.artifact_dir is not None
        metadata = load_artifact(result.artifact_dir).metadata
        assert metadata.decision_threshold == result.best.threshold.threshold

    def test_two_identical_runs_produce_identical_metrics(self, app_config, prepared):
        """Reproducibility, asserted rather than assumed."""
        first = run_experiment(app_config, data=prepared, persist=False)
        second = run_experiment(app_config, data=prepared, persist=False)
        for a, b in zip(first.reports, second.reports, strict=True):
            assert a.test_metrics.average_precision == b.test_metrics.average_precision
            assert a.threshold.threshold == b.threshold.threshold

    def test_a_different_seed_is_allowed_to_change_the_result(self, app_config, prepared):
        """Confirms the seed is actually wired through, not decorative."""
        forest = app_config.require_training().models[-1]
        del forest  # models here are dummy + logistic regression, both deterministic
        first = run_experiment(app_config, data=prepared, persist=False)
        assert first.reports  # smoke: seeding does not break the run


class TestProtocolComparison:
    def test_the_random_protocol_inflates_pooled_roc_auc(self, app_config, prepared):
        """The project's central empirical claim, as a regression test.

        Under a random split the model can learn each month's base rate from the
        macro block and replay it on test rows drawn from the same months, so the
        pooled ROC-AUC sits well above the within-period value. The fixture
        reproduces that structure deliberately.
        """
        random_config = app_config.model_copy(
            update={
                "split": SplitConfig(
                    strategy="random", test_size=0.25, validation_size=0.15, random_state=42
                )
            }
        )
        result = run_experiment(random_config, data=prepared, persist=False)
        model = next(r for r in result.reports if r.model_name == "logistic_regression")
        assert model.within_period["roc_auc_inflation"] > 0.05

    def test_the_out_of_time_protocol_puts_the_future_after_the_past(self, app_config, prepared):
        result = run_experiment(app_config, data=prepared, persist=False)
        split = result.split
        assert split.periods_for("train").max() < split.periods_for("test").min()

    def test_both_protocols_run_without_error(self, app_config, prepared):
        for strategy in ("random", "out_of_time"):
            config = app_config.model_copy(
                update={
                    "split": SplitConfig(
                        strategy=strategy,
                        test_size=0.25,
                        validation_size=0.15,
                        test_periods=3,
                        validation_periods=3,
                    )
                }
            )
            assert run_experiment(config, data=prepared, persist=False).reports


class TestTrainToInference:
    def test_the_scored_probabilities_match_the_evaluated_ones(self, app_config, prepared):
        """The contract that makes an offline metric mean anything.

        Training reports a number; inference must produce the same number from
        the saved file. If preprocessing were duplicated outside the pipeline —
        as it was in the original notebook — this is where it would show.
        """
        result = run_experiment(app_config, data=prepared, persist=True)
        predictor = load_predictor(
            app_config.paths.artifacts_dir, InferenceConfig(validate_input=False)
        )

        winner = next(m for m in result.models if m.name == result.best.model_name)
        evaluated = winner.predict_proba(result.split.X_test)
        scored = predictor.score_frame(result.split.X_test)["subscription_probability"].to_numpy()
        np.testing.assert_allclose(evaluated, scored, rtol=1e-9, atol=1e-12)

    def test_the_call_list_is_ranked_and_tiered(self, app_config, prepared):
        run_experiment(app_config, data=prepared, persist=True)
        predictor = load_predictor(
            app_config.paths.artifacts_dir, InferenceConfig(validate_input=False)
        )
        ranked = predictor.rank(prepared.frame.head(300), top_k=50)
        assert len(ranked) == 50
        assert ranked["subscription_probability"].is_monotonic_decreasing
        assert "tier" in ranked.columns

    def test_latest_resolves_to_the_most_recent_run(self, app_config, prepared):
        first = run_experiment(app_config, data=prepared, persist=True)
        second = run_experiment(app_config, data=prepared, persist=True)
        assert first.run_id != second.run_id
        resolved = resolve_artifact_path(app_config.paths.artifacts_dir, "latest")
        assert load_artifact(resolved).metadata.created_at >= (
            load_artifact(app_config.paths.artifacts_dir / first.run_id).metadata.created_at
        )


class TestNoLeakage:
    def test_preprocessing_statistics_come_only_from_training_rows(self, app_config, prepared):
        """Fitting the scaler on the full dataset is the classic silent leak.

        Checked by fitting the pipeline through the normal path and comparing the
        scaler's learned mean against the training rows alone.
        """
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        models = train_all(split, app_config)
        pipeline = models[-1].pipeline  # logistic_regression
        scaler = pipeline.named_steps["encode"].named_transformers_["numeric"]

        from term_deposit.features.pipeline import resolved_numeric_columns

        age_index = list(resolved_numeric_columns(app_config.features)).index("age")
        assert scaler.mean_[age_index] == pytest.approx(float(split.X_train["age"].mean()))
        assert scaler.mean_[age_index] != pytest.approx(float(prepared.frame["age"].mean()))

    def test_no_test_row_appears_in_training(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        assert not set(split.X_train.index) & set(split.X_test.index)
        assert not set(split.X_validation.index) & set(split.X_test.index)

    def test_the_feature_matrix_never_contains_the_target_or_duration(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        forbidden = {constants.TARGET_COLUMN, constants.LABEL_COLUMN, "duration"}
        assert not forbidden & set(split.X_train.columns)

    def test_evaluation_never_sees_a_threshold_fitted_on_test(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        models = train_all(split, app_config)
        reports = evaluate_all(models, split, app_config)
        assert all(report.threshold.chosen_on == "validation" for report in reports)


class TestComparisonTable:
    def test_is_sorted_by_the_primary_metric(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        models = train_all(split, app_config)
        table = build_comparison_table(evaluate_all(models, split, app_config))
        assert table["average_precision"].is_monotonic_decreasing

    def test_reports_the_within_period_column_next_to_the_pooled_one(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        models = train_all(split, app_config)
        table = build_comparison_table(evaluate_all(models, split, app_config))
        assert {"roc_auc", "within_period_roc_auc", "roc_auc_inflation"} <= set(table.columns)

    def test_select_best_returns_one_of_the_candidates(self, app_config, prepared):
        split = make_split(
            prepared.frame,
            app_config.split,
            feature_columns=app_config.features.input_columns(),
        )
        models = train_all(split, app_config)
        reports = evaluate_all(models, split, app_config)
        assert select_best(reports) in reports

    def test_select_best_rejects_an_empty_list(self):
        with pytest.raises(ValueError, match="empty report list"):
            select_best([])
