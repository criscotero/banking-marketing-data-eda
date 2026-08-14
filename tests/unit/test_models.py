"""Estimator construction and artifact persistence.

The persistence tests are the ones that matter operationally: an artifact that
loads but scores differently from the model that was evaluated is the worst
failure this system can have, because nothing about it looks broken.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from term_deposit.config import FeatureConfig, ModelSpec
from term_deposit.models.persistence import (
    ArtifactError,
    ModelArtifact,
    ModelMetadata,
    list_artifacts,
    load_artifact,
    resolve_artifact_path,
    save_artifact,
)
from term_deposit.models.registry import (
    MODEL_STEP,
    UnknownEstimatorError,
    build_estimator,
    build_pipeline,
    registered_estimators,
)


class TestBuildEstimator:
    def test_builds_each_registered_estimator(self):
        for name in registered_estimators():
            spec = ModelSpec(
                name=name,
                estimator=name,
                balance_strategy="scale_pos_weight" if name == "xgboost" else "none",
            )
            assert build_estimator(spec, random_state=42, scale_pos_weight=8.0) is not None

    def test_rejects_an_unregistered_estimator(self):
        spec = ModelSpec(name="x", estimator="dummy")
        object.__setattr__(spec, "estimator", "not_a_model")
        with pytest.raises(UnknownEstimatorError, match="registered estimators"):
            build_estimator(spec, random_state=42)

    def test_applies_the_seed_to_estimators_that_accept_one(self):
        spec = ModelSpec(name="rf", estimator="random_forest")
        assert build_estimator(spec, random_state=7).random_state == 7

    def test_does_not_seed_the_deterministic_baseline(self):
        """DummyClassifier(strategy='prior') has no randomness to seed."""
        spec = ModelSpec(name="d", estimator="dummy")
        assert build_estimator(spec, random_state=7).get_params().get("random_state") is None

    def test_explicit_params_beat_defaults(self):
        spec = ModelSpec(name="rf", estimator="random_forest", params={"n_estimators": 11})
        assert build_estimator(spec, random_state=42).n_estimators == 11

    def test_class_weight_strategy_sets_balanced(self):
        spec = ModelSpec(
            name="lr", estimator="logistic_regression", balance_strategy="class_weight"
        )
        assert build_estimator(spec, random_state=42).class_weight == "balanced"

    def test_scale_pos_weight_strategy_requires_the_ratio(self):
        """It depends on the training fold, so it cannot come from YAML."""
        spec = ModelSpec(name="xgb", estimator="xgboost", balance_strategy="scale_pos_weight")
        with pytest.raises(ValueError, match="no training-split ratio"):
            build_estimator(spec, random_state=42, scale_pos_weight=None)

    def test_scale_pos_weight_is_forwarded(self):
        spec = ModelSpec(name="xgb", estimator="xgboost", balance_strategy="scale_pos_weight")
        estimator = build_estimator(spec, random_state=42, scale_pos_weight=7.5)
        assert estimator.get_params()["scale_pos_weight"] == 7.5

    def test_n_jobs_is_not_passed_to_estimators_that_deprecated_it(self, labelled_frame):
        """scikit-learn 1.8 deprecated n_jobs on LogisticRegression.

        Asserted by fitting under ``-W error`` rather than by inspecting params,
        because the parameter still exists — it just warns when set.
        """
        spec = ModelSpec(name="lr", estimator="logistic_regression", params={"max_iter": 50})
        pipeline = build_pipeline(spec, FeatureConfig(), random_state=42, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            pipeline.fit(labelled_frame.head(400), labelled_frame["subscribed"].head(400))

    def test_n_jobs_is_passed_to_estimators_that_support_it(self):
        spec = ModelSpec(name="rf", estimator="random_forest")
        assert build_estimator(spec, random_state=42, n_jobs=3).n_jobs == 3


class TestBuildPipeline:
    def test_the_estimator_is_the_final_step(self):
        spec = ModelSpec(name="lr", estimator="logistic_regression")
        pipeline = build_pipeline(spec, FeatureConfig(), random_state=42)
        assert pipeline.steps[-1][0] == MODEL_STEP
        assert isinstance(pipeline.named_steps[MODEL_STEP], LogisticRegression)

    def test_each_pipeline_gets_its_own_preprocessor(self):
        spec = ModelSpec(name="lr", estimator="logistic_regression")
        first = build_pipeline(spec, FeatureConfig(), random_state=42)
        second = build_pipeline(spec, FeatureConfig(), random_state=42)
        assert first.named_steps["encode"] is not second.named_steps["encode"]

    def test_preprocessing_travels_with_the_model(self, labelled_frame):
        """The property that keeps training and inference identical."""
        spec = ModelSpec(name="rf", estimator="random_forest", params={"n_estimators": 5})
        pipeline = build_pipeline(spec, FeatureConfig(), random_state=42)
        pipeline.fit(labelled_frame, labelled_frame["subscribed"])
        # Raw columns in, probabilities out: no external transformation needed.
        assert pipeline.predict_proba(labelled_frame.head(3)).shape == (3, 2)


class TestModelMetadata:
    def test_create_stamps_the_environment(self):
        metadata = ModelMetadata.create(
            model_name="m",
            estimator="dummy",
            split_strategy="out_of_time",
            feature_set="all",
            input_columns=("age",),
            decision_threshold=0.5,
            threshold_objective="expected_value",
        )
        assert metadata.created_at
        assert "scikit-learn" in metadata.library_versions
        assert "python" in metadata.library_versions

    def test_round_trips_through_a_dict(self):
        original = ModelMetadata.create(
            model_name="m",
            estimator="dummy",
            split_strategy="random",
            feature_set="client_only",
            input_columns=("age", "job"),
            decision_threshold=0.31,
            threshold_objective="f1",
            notes=("a caveat",),
        )
        restored = ModelMetadata.from_dict(original.to_dict())
        assert restored.input_columns == ("age", "job")
        assert restored.notes == ("a caveat",)
        assert restored.decision_threshold == 0.31

    def test_the_dict_form_is_json_serialisable(self):
        metadata = ModelMetadata.create(
            model_name="m",
            estimator="dummy",
            split_strategy="random",
            feature_set="all",
            input_columns=("age",),
            decision_threshold=0.5,
            threshold_objective="f1",
        )
        assert json.loads(json.dumps(metadata.to_dict()))["model_name"] == "m"


@pytest.fixture
def fitted_artifact(labelled_frame) -> ModelArtifact:
    """A trained pipeline plus metadata, ready to persist."""
    spec = ModelSpec(name="rf", estimator="random_forest", params={"n_estimators": 8})
    features = FeatureConfig()
    pipeline = build_pipeline(spec, features, random_state=42)
    pipeline.fit(labelled_frame, labelled_frame["subscribed"])
    metadata = ModelMetadata.create(
        model_name="rf",
        estimator="random_forest",
        split_strategy="out_of_time",
        feature_set="all",
        input_columns=features.input_columns(),
        decision_threshold=0.27,
        threshold_objective="expected_value",
        metrics={"test": {"average_precision": 0.68}},
    )
    return ModelArtifact(pipeline=pipeline, metadata=metadata)


class TestPersistence:
    def test_saving_then_loading_preserves_predictions_exactly(
        self, fitted_artifact, labelled_frame, tmp_path
    ):
        """The check that the shipped file *is* the evaluated model."""
        directory = save_artifact(fitted_artifact, tmp_path / "run1")
        before = fitted_artifact.pipeline.predict_proba(labelled_frame.head(50))[:, 1]
        after = load_artifact(directory).pipeline.predict_proba(labelled_frame.head(50))[:, 1]
        np.testing.assert_array_equal(before, after)

    def test_metadata_survives_the_round_trip(self, fitted_artifact, tmp_path):
        directory = save_artifact(fitted_artifact, tmp_path / "run1")
        loaded = load_artifact(directory)
        assert loaded.metadata.decision_threshold == 0.27
        assert loaded.metadata.split_strategy == "out_of_time"
        assert loaded.metadata.input_columns == fitted_artifact.metadata.input_columns

    def test_writes_both_files(self, fitted_artifact, tmp_path):
        directory = save_artifact(fitted_artifact, tmp_path / "run1")
        assert (directory / "model.joblib").is_file()
        assert (directory / "metadata.json").is_file()

    def test_latest_resolves_to_the_most_recent_run(self, fitted_artifact, tmp_path):
        save_artifact(fitted_artifact, tmp_path / "run1")
        save_artifact(fitted_artifact, tmp_path / "run2")
        assert resolve_artifact_path(tmp_path).resolve() == (tmp_path / "run2").resolve()

    def test_a_named_run_can_be_pinned(self, fitted_artifact, tmp_path):
        save_artifact(fitted_artifact, tmp_path / "run1")
        save_artifact(fitted_artifact, tmp_path / "run2")
        assert resolve_artifact_path(tmp_path, "run1") == tmp_path / "run1"

    def test_an_unknown_run_id_is_rejected(self, fitted_artifact, tmp_path):
        save_artifact(fitted_artifact, tmp_path / "run1")
        with pytest.raises(ArtifactError, match="no model artifact"):
            resolve_artifact_path(tmp_path, "run99")

    def test_an_empty_directory_names_the_fix(self, tmp_path):
        with pytest.raises(ArtifactError, match=r"scripts/train.py"):
            resolve_artifact_path(tmp_path / "empty")

    def test_list_artifacts_excludes_the_latest_pointer(self, fitted_artifact, tmp_path):
        save_artifact(fitted_artifact, tmp_path / "run1")
        save_artifact(fitted_artifact, tmp_path / "run2")
        assert {path.name for path in list_artifacts(tmp_path)} == {"run1", "run2"}

    def test_list_artifacts_on_a_missing_directory_returns_empty(self, tmp_path):
        assert list_artifacts(tmp_path / "nope") == []

    def test_loading_without_metadata_is_rejected(self, fitted_artifact, tmp_path):
        directory = save_artifact(fitted_artifact, tmp_path / "run1", update_latest=False)
        (directory / "metadata.json").unlink()
        with pytest.raises(ArtifactError, match="metadata file not found"):
            load_artifact(directory)

    def test_loading_without_a_model_is_rejected(self, fitted_artifact, tmp_path):
        directory = save_artifact(fitted_artifact, tmp_path / "run1", update_latest=False)
        (directory / "model.joblib").unlink()
        with pytest.raises(ArtifactError, match="model file not found"):
            load_artifact(directory)

    def test_a_corrupt_model_file_explains_the_version_risk(self, fitted_artifact, tmp_path):
        directory = save_artifact(fitted_artifact, tmp_path / "run1", update_latest=False)
        (directory / "model.joblib").write_bytes(b"not a pickle")
        with pytest.raises(ArtifactError, match="library versions"):
            load_artifact(directory)

    def test_predict_proba_returns_the_positive_column(self, fitted_artifact, labelled_frame):
        probabilities = fitted_artifact.predict_proba(labelled_frame.head(10))
        assert probabilities.shape == (10,)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
