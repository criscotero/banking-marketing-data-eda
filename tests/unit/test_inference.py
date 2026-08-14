"""The inference contract and the scoring surface.

The contract's job is to reject bad input loudly. A scorer that silently accepts
a mis-typed column produces a plausible ranking that is wrong, which is worse
than an outage because nobody notices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from term_deposit.config import FeatureConfig, InferenceConfig, ModelSpec
from term_deposit.inference.predictor import Predictor, PredictorError, load_predictor
from term_deposit.inference.schema import (
    CustomerRecord,
    ScoringRequest,
    validate_frame,
)
from term_deposit.models.persistence import ModelArtifact, ModelMetadata, save_artifact
from term_deposit.models.registry import build_pipeline

VALID_RECORD = {
    "age": 41,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1,
}


class TestCustomerRecord:
    def test_accepts_a_valid_record(self):
        assert CustomerRecord.model_validate(VALID_RECORD).age == 41

    def test_rejects_the_post_call_field(self):
        """``duration`` is only known after the call.

        Accepting it would let a caller feed the outcome into the model that
        exists to predict that outcome.
        """
        with pytest.raises(Exception, match=r"extra_forbidden|Extra inputs"):
            CustomerRecord.model_validate({**VALID_RECORD, "duration": 300})

    def test_rejects_the_target_column(self):
        with pytest.raises(Exception, match=r"extra_forbidden|Extra inputs"):
            CustomerRecord.model_validate({**VALID_RECORD, "y": "yes"})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("age", 5),
            ("age", 200),
            ("job", "astronaut"),
            ("marital", "widowed"),
            ("month", "smarch"),
            ("day_of_week", "sun"),
            ("campaign", 0),
            ("pdays", 1500),
            ("previous", -1),
            ("poutcome", "maybe"),
            ("euribor3m", -1.0),
        ],
    )
    def test_rejects_out_of_domain_values(self, field, value):
        with pytest.raises(ValidationError):
            CustomerRecord.model_validate({**VALID_RECORD, field: value})

    def test_accepts_the_never_contacted_sentinel(self):
        assert CustomerRecord.model_validate({**VALID_RECORD, "pdays": 999}).pdays == 999

    def test_to_row_uses_the_raw_dotted_column_names(self):
        row = CustomerRecord.model_validate(VALID_RECORD).to_row()
        assert "emp.var.rate" in row
        assert "nr.employed" in row
        assert "customer_id" not in row

    def test_is_immutable(self):
        record = CustomerRecord.model_validate(VALID_RECORD)
        with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
            record.age = 50  # type: ignore[misc]


class TestScoringRequest:
    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValidationError):
            ScoringRequest(records=[])

    def test_to_frame_produces_raw_column_names(self):
        request = ScoringRequest(records=[CustomerRecord.model_validate(VALID_RECORD)])
        frame = request.to_frame()
        assert "cons.price.idx" in frame.columns
        assert len(frame) == 1

    def test_customer_ids_stay_positionally_aligned(self):
        request = ScoringRequest(
            records=[
                CustomerRecord.model_validate({**VALID_RECORD, "customer_id": "a"}),
                CustomerRecord.model_validate({**VALID_RECORD, "customer_id": "b"}),
            ]
        )
        assert request.customer_ids() == ["a", "b"]


class TestValidateFrame:
    def test_returns_the_required_columns_in_model_order(self):
        frame = pd.DataFrame([VALID_RECORD, VALID_RECORD])
        required = ("age", "job", "euribor3m")
        assert list(validate_frame(frame, required_columns=required).columns) == list(required)

    def test_names_the_missing_columns(self):
        frame = pd.DataFrame([VALID_RECORD]).drop(columns=["euribor3m"])
        with pytest.raises(ValueError, match="missing required column"):
            validate_frame(frame, required_columns=("age", "euribor3m"))

    def test_reports_the_offending_row_number(self):
        frame = pd.DataFrame([VALID_RECORD, {**VALID_RECORD, "job": "astronaut"}])
        with pytest.raises(ValueError, match="row 1"):
            validate_frame(frame, required_columns=("age", "job"))

    def test_reports_several_bad_rows_at_once(self):
        rows = [VALID_RECORD] + [{**VALID_RECORD, "age": 999} for _ in range(3)]
        with pytest.raises(ValueError) as info:
            validate_frame(pd.DataFrame(rows), required_columns=("age",))
        assert str(info.value).count("row ") >= 3


@pytest.fixture
def artifact_dir(tmp_path, labelled_frame):
    """A persisted model, ready for the predictor to load."""
    features = FeatureConfig()
    spec = ModelSpec(name="rf", estimator="random_forest", params={"n_estimators": 8})
    pipeline = build_pipeline(spec, features, random_state=42)
    pipeline.fit(labelled_frame, labelled_frame["subscribed"])
    metadata = ModelMetadata.create(
        model_name="rf",
        estimator="random_forest",
        split_strategy="out_of_time",
        feature_set="all",
        input_columns=features.input_columns(),
        decision_threshold=0.3,
        threshold_objective="expected_value",
        notes=("a caveat worth surfacing",),
    )
    save_artifact(ModelArtifact(pipeline=pipeline, metadata=metadata), tmp_path / "run1")
    return tmp_path


class TestPredictor:
    def test_loads_from_the_artifacts_directory(self, artifact_dir):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        assert predictor.artifact.metadata.model_name == "rf"

    def test_uses_the_threshold_recorded_in_the_artifact(self, artifact_dir):
        """Not 0.5. The shipped threshold is the one chosen on validation."""
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        assert predictor.threshold == 0.3

    def test_predicted_class_follows_that_threshold(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        scored = predictor.score_frame(labelled_frame.head(200))
        expected = (scored["subscription_probability"] >= 0.3).astype(int)
        assert (scored["predicted_class"] == expected).all()

    def test_scores_are_probabilities(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        scored = predictor.score_frame(labelled_frame.head(100))
        assert scored["subscription_probability"].between(0, 1).all()

    def test_batching_does_not_change_the_result(self, artifact_dir, labelled_frame):
        frame = labelled_frame.head(300)
        big = load_predictor(artifact_dir, InferenceConfig(validate_input=False, batch_size=10_000))
        small = load_predictor(artifact_dir, InferenceConfig(validate_input=False, batch_size=7))
        np.testing.assert_allclose(
            big.score_frame(frame)["subscription_probability"].to_numpy(),
            small.score_frame(frame)["subscription_probability"].to_numpy(),
        )

    def test_extra_columns_are_ignored(self, artifact_dir, labelled_frame):
        """A CRM export carries more than the model needs."""
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        frame = labelled_frame.head(50)
        with_extra = frame.assign(crm_notes="anything", segment="x")
        np.testing.assert_allclose(
            predictor.score_frame(frame)["subscription_probability"].to_numpy(),
            predictor.score_frame(with_extra)["subscription_probability"].to_numpy(),
        )

    def test_column_order_does_not_change_the_result(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        frame = labelled_frame.head(50)
        reordered = frame[list(reversed(frame.columns))]
        np.testing.assert_allclose(
            predictor.score_frame(frame)["subscription_probability"].to_numpy(),
            predictor.score_frame(reordered)["subscription_probability"].to_numpy(),
        )

    def test_missing_required_columns_are_rejected(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        with pytest.raises((KeyError, ValueError)):
            predictor.score_frame(labelled_frame.head(10).drop(columns=["euribor3m"]))

    def test_validation_rejects_an_out_of_domain_value(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=True))
        frame = labelled_frame.head(10).copy()
        frame.loc[frame.index[0], "job"] = "astronaut"
        with pytest.raises(ValueError, match="failed validation"):
            predictor.score_frame(frame)

    def test_an_empty_frame_is_rejected(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        with pytest.raises(PredictorError, match="empty frame"):
            predictor.score_frame(labelled_frame.head(0))

    def test_tiers_are_attached_when_configured(self, artifact_dir, labelled_frame):
        predictor = load_predictor(
            artifact_dir, InferenceConfig(validate_input=False, include_tier=True)
        )
        scored = predictor.score_frame(labelled_frame.head(200))
        assert set(scored["tier"].unique()) <= set(InferenceConfig().tier_labels)

    def test_rank_returns_a_descending_call_list(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        ranked = predictor.rank(labelled_frame.head(200))
        assert ranked["subscription_probability"].is_monotonic_decreasing
        assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))

    def test_rank_honours_top_k(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        assert len(predictor.rank(labelled_frame.head(200), top_k=25)) == 25

    def test_an_id_column_is_carried_through(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        frame = labelled_frame.head(20).assign(crm_id=[f"c{i}" for i in range(20)])
        scored = predictor.score_frame(frame, id_column="crm_id")
        assert scored["customer_id"].tolist() == [f"c{i}" for i in range(20)]

    def test_an_unknown_id_column_is_rejected(self, artifact_dir, labelled_frame):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        with pytest.raises(PredictorError, match="not found"):
            predictor.score_frame(labelled_frame.head(5), id_column="nope")

    def test_score_request_returns_a_typed_response(self, artifact_dir):
        predictor = load_predictor(artifact_dir, InferenceConfig(validate_input=False))
        request = ScoringRequest(
            records=[
                CustomerRecord.model_validate({**VALID_RECORD, "customer_id": "a"}),
                CustomerRecord.model_validate({**VALID_RECORD, "age": 70, "customer_id": "b"}),
            ]
        )
        response = predictor.score_request(request)
        assert response.model_name == "rf"
        assert [p.customer_id for p in response.predictions] == ["a", "b"]
        assert response.decision_threshold == 0.3

    def test_an_artifact_without_input_columns_is_rejected(self, artifact_dir):
        """Without a recorded contract there is nothing to validate against."""
        from term_deposit.models.persistence import load_artifact

        artifact = load_artifact(artifact_dir / "run1")
        stripped = ModelArtifact(
            pipeline=artifact.pipeline,
            metadata=ModelMetadata.from_dict({**artifact.metadata.to_dict(), "input_columns": []}),
        )
        with pytest.raises(PredictorError, match="input columns"):
            Predictor(stripped)

    def test_a_missing_artifact_directory_names_the_fix(self, tmp_path):
        with pytest.raises(PredictorError, match=r"scripts/train.py"):
            load_predictor(tmp_path / "empty")
