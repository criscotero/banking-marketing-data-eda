"""Feature transformations and the preprocessing pipeline.

The pipeline is the contract between training and inference, so these tests
concentrate on the properties that keep the two identical: no shared mutable
state, no fitting on data the model should not see, and stable output ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from term_deposit import constants
from term_deposit.config import FeatureConfig
from term_deposit.features.pipeline import (
    build_preprocessor,
    output_feature_names,
    resolved_numeric_columns,
)
from term_deposit.features.transformers import ColumnSelector, PdaysSentinelEncoder


class TestPdaysSentinelEncoder:
    def test_splits_the_sentinel_into_a_flag_and_a_duration(self):
        frame = pd.DataFrame({"pdays": [999, 3, 999, 12], "age": [40, 51, 33, 29]})
        result = PdaysSentinelEncoder().fit_transform(frame)
        assert result["pdays_never_contacted"].tolist() == [1, 0, 1, 0]
        assert result["pdays_days_since_contact"].tolist() == [0.0, 3.0, 0.0, 12.0]

    def test_removes_the_original_column(self):
        frame = pd.DataFrame({"pdays": [999], "age": [40]})
        assert "pdays" not in PdaysSentinelEncoder().fit_transform(frame).columns

    def test_keeps_the_new_columns_in_the_original_position(self):
        frame = pd.DataFrame({"age": [40], "pdays": [999], "campaign": [1]})
        columns = list(PdaysSentinelEncoder().fit_transform(frame).columns)
        assert columns == [
            "age",
            "pdays_never_contacted",
            "pdays_days_since_contact",
            "campaign",
        ]

    def test_does_not_mutate_the_input(self):
        frame = pd.DataFrame({"pdays": [999], "age": [40]})
        before = frame.copy()
        PdaysSentinelEncoder().fit_transform(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_is_stateless_across_fits(self):
        """The rule is fixed, so a value unseen during fit still encodes correctly.

        A learned imputation would silently place every unseen value at the
        training set's mean; this transformer cannot.
        """
        encoder = PdaysSentinelEncoder().fit(pd.DataFrame({"pdays": [999, 999]}))
        result = encoder.transform(pd.DataFrame({"pdays": [7]}))
        assert result["pdays_never_contacted"].tolist() == [0]
        assert result["pdays_days_since_contact"].tolist() == [7.0]

    def test_rejects_a_frame_without_the_column(self):
        with pytest.raises(ValueError, match="expects a 'pdays' column"):
            PdaysSentinelEncoder().fit(pd.DataFrame({"age": [40]}))

    def test_reports_output_feature_names(self):
        encoder = PdaysSentinelEncoder().fit(pd.DataFrame({"age": [1], "pdays": [999]}))
        assert list(encoder.get_feature_names_out()) == [
            "age",
            "pdays_never_contacted",
            "pdays_days_since_contact",
        ]

    def test_requires_fitting_before_transform(self):
        with pytest.raises(NotFittedError):
            PdaysSentinelEncoder().transform(pd.DataFrame({"pdays": [999]}))


class TestColumnSelector:
    def test_selects_and_reorders(self):
        frame = pd.DataFrame({"b": [1], "a": [2], "c": [3]})
        result = ColumnSelector(("a", "b")).fit_transform(frame)
        assert list(result.columns) == ["a", "b"]

    def test_drops_extra_columns(self):
        """A scoring frame carrying extra columns must score identically."""
        selector = ColumnSelector(("a", "b")).fit(pd.DataFrame({"a": [1], "b": [2]}))
        result = selector.transform(pd.DataFrame({"a": [1], "b": [2], "unexpected": [9]}))
        assert list(result.columns) == ["a", "b"]

    def test_rejects_missing_columns_at_fit_time(self):
        with pytest.raises(ValueError, match="missing required column"):
            ColumnSelector(("a", "b")).fit(pd.DataFrame({"a": [1]}))

    def test_rejects_missing_columns_at_transform_time(self):
        selector = ColumnSelector(("a", "b")).fit(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match="missing required column"):
            selector.transform(pd.DataFrame({"a": [1]}))


class TestResolvedNumericColumns:
    def test_expands_pdays_when_the_encoder_is_enabled(self):
        columns = resolved_numeric_columns(FeatureConfig(encode_pdays_sentinel=True))
        assert "pdays" not in columns
        assert "pdays_never_contacted" in columns
        assert "pdays_days_since_contact" in columns

    def test_leaves_pdays_alone_when_disabled(self):
        columns = resolved_numeric_columns(FeatureConfig(encode_pdays_sentinel=False))
        assert "pdays" in columns

    def test_client_only_drops_the_macro_block(self):
        columns = resolved_numeric_columns(FeatureConfig(feature_set="client_only"))
        assert not set(columns) & set(constants.MACRO_FEATURES)


class TestBuildPreprocessor:
    def test_returns_a_new_instance_every_call(self):
        """Each pipeline must own its preprocessor.

        The original notebook shared one ``ColumnTransformer`` across four
        pipelines, so a later ``fit_transform`` refitted the transformer inside
        every already-trained model.
        """
        config = FeatureConfig()
        first, second = build_preprocessor(config), build_preprocessor(config)
        assert first is not second
        assert first.named_steps["encode"] is not second.named_steps["encode"]

    def test_fitting_one_does_not_fit_the_other(self, labelled_frame):
        config = FeatureConfig()
        first, second = build_preprocessor(config), build_preprocessor(config)
        first.fit(labelled_frame)
        with pytest.raises((NotFittedError, RuntimeError)):
            output_feature_names(second)

    def test_output_is_numeric_and_finite(self, labelled_frame):
        matrix = build_preprocessor(FeatureConfig()).fit_transform(labelled_frame)
        assert np.isfinite(np.asarray(matrix, dtype=float)).all()

    def test_feature_names_match_the_matrix_width(self, labelled_frame):
        preprocessor = build_preprocessor(FeatureConfig())
        matrix = preprocessor.fit_transform(labelled_frame)
        assert len(output_feature_names(preprocessor)) == matrix.shape[1]

    def test_unseen_categories_do_not_raise(self, labelled_frame):
        """A new job title must not take the scorer down."""
        preprocessor = build_preprocessor(FeatureConfig()).fit(labelled_frame)
        unseen = labelled_frame.head(3).copy()
        unseen["job"] = "astronaut"
        matrix = preprocessor.transform(unseen)
        assert matrix.shape[1] == len(output_feature_names(preprocessor))

    def test_unseen_categories_encode_as_an_all_zero_block(self, labelled_frame):
        preprocessor = build_preprocessor(FeatureConfig()).fit(labelled_frame)
        names = output_feature_names(preprocessor)
        unseen = labelled_frame.head(1).copy()
        unseen["job"] = "astronaut"
        matrix = np.asarray(preprocessor.transform(unseen))
        job_columns = [index for index, name in enumerate(names) if name.startswith("job_")]
        assert matrix[0, job_columns].sum() == 0.0

    def test_column_order_does_not_affect_the_result(self, labelled_frame):
        preprocessor = build_preprocessor(FeatureConfig()).fit(labelled_frame)
        original = np.asarray(preprocessor.transform(labelled_frame.head(5)))
        shuffled = labelled_frame.head(5)[list(reversed(labelled_frame.columns))]
        np.testing.assert_allclose(original, np.asarray(preprocessor.transform(shuffled)))

    def test_scaling_uses_training_statistics_only(self, labelled_frame):
        """Fitting the scaler on the full dataset is the classic silent leak."""
        train = labelled_frame.head(500)
        preprocessor = build_preprocessor(FeatureConfig()).fit(train)
        scaler = preprocessor.named_steps["encode"].named_transformers_["numeric"]
        age_index = list(resolved_numeric_columns(FeatureConfig())).index("age")
        assert scaler.mean_[age_index] == pytest.approx(float(train["age"].mean()))

    def test_client_only_produces_fewer_columns(self, labelled_frame):
        full = build_preprocessor(FeatureConfig()).fit(labelled_frame)
        client = build_preprocessor(FeatureConfig(feature_set="client_only")).fit(labelled_frame)
        assert len(output_feature_names(client)) < len(output_feature_names(full))
        assert not set(output_feature_names(client)) & set(constants.MACRO_FEATURES)

    def test_unfitted_preprocessor_reports_a_clear_error(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            output_feature_names(build_preprocessor(FeatureConfig()))
