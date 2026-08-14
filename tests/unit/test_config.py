"""Configuration loading, merging and validation.

Configuration is the project's control surface, so the tests focus on the ways a
config can be silently wrong: a typo that is ignored, a layer that fails to
override, a contradictory combination that validates anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from term_deposit import constants
from term_deposit.config import (
    AppConfig,
    ConfigError,
    FeatureConfig,
    InferenceConfig,
    ModelSpec,
    PathsConfig,
    SplitConfig,
    TrainingConfig,
    apply_overrides,
    deep_merge,
    load_config,
    parse_override,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class TestDeepMerge:
    def test_merges_nested_mappings_key_by_key(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_replaces_lists_wholesale(self):
        """So that a layer can shorten a model list, not only extend it."""
        merged = deep_merge({"models": [1, 2, 3]}, {"models": [9]})
        assert merged == {"models": [9]}

    def test_does_not_mutate_its_inputs(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestParseOverride:
    def test_parses_a_dotted_key(self):
        path, value = parse_override("split.strategy=random")
        assert list(path) == ["split", "strategy"]
        assert value == "random"

    def test_parses_values_as_yaml_so_types_survive(self):
        assert parse_override("training.cv_folds=3")[1] == 3
        assert parse_override("training.backtest=false")[1] is False
        assert parse_override("evaluation.top_k_fractions=[0.1, 0.2]")[1] == [0.1, 0.2]

    def test_rejects_a_missing_equals_sign(self):
        with pytest.raises(ConfigError, match="not of the form"):
            parse_override("split.strategy")

    def test_rejects_an_empty_key(self):
        with pytest.raises(ConfigError, match="empty key"):
            parse_override("=random")


class TestApplyOverrides:
    def test_creates_intermediate_levels(self):
        assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}

    def test_later_overrides_win(self):
        result = apply_overrides({}, ["a=1", "a=2"])
        assert result == {"a": 2}


class TestAppConfig:
    def test_rejects_unknown_keys(self):
        """A typo must fail loudly.

        Silently ignoring ``random_seed`` (instead of ``random_state``) would
        produce an unreproducible run that looks perfectly fine.
        """
        with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
            AppConfig.model_validate({"random_seed": 1})

    def test_is_immutable(self):
        config = AppConfig()
        with pytest.raises(ValidationError, match=r"frozen|Instance is frozen"):
            # mypy already knows this is read-only; the test pins the *runtime*
            # behaviour, so that a config cannot be mutated after a run starts.
            config.random_state = 99  # type: ignore[misc]

    def test_require_training_explains_what_is_missing(self):
        with pytest.raises(ConfigError, match=r"configs/training.yaml"):
            AppConfig().require_training()


class TestSplitConfig:
    def test_rejects_fractions_that_leave_no_training_rows(self):
        with pytest.raises(ValueError, match="leave room"):
            SplitConfig(strategy="random", test_size=0.7, validation_size=0.4)

    def test_allows_the_same_fractions_under_the_temporal_protocol(self):
        """They are unused there, so they must not block a valid config."""
        assert SplitConfig(strategy="out_of_time", test_size=0.7, validation_size=0.4)

    @pytest.mark.parametrize("size", [0.0, 1.0, -0.1])
    def test_rejects_out_of_range_test_sizes(self, size):
        with pytest.raises(ValueError):
            SplitConfig(test_size=size)


class TestFeatureConfig:
    def test_all_includes_the_macro_block(self):
        assert set(constants.MACRO_FEATURES) <= set(FeatureConfig().numeric_columns())

    def test_client_only_excludes_it(self):
        columns = FeatureConfig(feature_set="client_only").numeric_columns()
        assert not set(constants.MACRO_FEATURES) & set(columns)

    def test_input_columns_are_numeric_then_categorical(self):
        config = FeatureConfig()
        assert config.input_columns() == (
            *config.numeric_columns(),
            *config.categorical_columns(),
        )

    def test_input_columns_never_include_the_post_call_field(self):
        """`duration` is the leak the whole project is built to avoid."""
        for feature_set in ("all", "client_only"):
            columns = FeatureConfig(feature_set=feature_set).input_columns()
            assert not set(constants.POST_OUTCOME_COLUMNS) & set(columns)


class TestTrainingConfig:
    def test_rejects_duplicate_model_names(self):
        """Duplicates would silently overwrite each other's report files."""
        spec = ModelSpec(name="dup", estimator="dummy")
        with pytest.raises(ValueError, match="duplicate model names"):
            TrainingConfig(models=(spec, spec))

    def test_rejects_a_config_that_enables_nothing(self):
        with pytest.raises(ValueError, match="enables no models"):
            TrainingConfig(models=(ModelSpec(name="a", estimator="dummy", enabled=False),))

    def test_enabled_models_filters_disabled_entries(self):
        config = TrainingConfig(
            models=(
                ModelSpec(name="on", estimator="dummy"),
                ModelSpec(name="off", estimator="dummy", enabled=False),
            )
        )
        assert [spec.name for spec in config.enabled_models] == ["on"]


class TestInferenceConfig:
    def test_requires_one_more_label_than_boundary(self):
        with pytest.raises(ValueError, match="one more entry"):
            InferenceConfig(tier_quantiles=(0.1, 0.5), tier_labels=("a", "b"))

    def test_requires_ascending_boundaries(self):
        with pytest.raises(ValueError, match="ascending"):
            InferenceConfig(tier_quantiles=(0.5, 0.1), tier_labels=("a", "b", "c"))


class TestPathsConfig:
    def test_resolve_anchors_relative_paths(self, tmp_path):
        resolved = PathsConfig().resolve(tmp_path)
        assert resolved.raw_dir == tmp_path / "data" / "raw"
        assert resolved.raw_csv.name == constants.RAW_FILENAME

    def test_resolve_leaves_absolute_paths_alone(self, tmp_path):
        absolute = tmp_path / "elsewhere"
        resolved = PathsConfig(raw_dir=absolute).resolve(tmp_path / "root")
        assert resolved.raw_dir == absolute

    def test_ensure_directories_creates_report_subdirectories(self, tmp_path):
        config = PathsConfig().resolve(tmp_path)
        config.ensure_directories()
        assert config.figures_dir.is_dir()
        assert config.metrics_dir.is_dir()


class TestLoadConfig:
    def test_layers_files_left_to_right(self, tmp_path):
        (tmp_path / "a.yaml").write_text(yaml.safe_dump({"random_state": 1, "log_level": "INFO"}))
        (tmp_path / "b.yaml").write_text(yaml.safe_dump({"random_state": 2}))
        config = load_config([tmp_path / "a.yaml", tmp_path / "b.yaml"], project_root=tmp_path)
        assert config.random_state == 2
        assert config.log_level == "INFO"

    def test_overrides_beat_files(self, tmp_path):
        (tmp_path / "a.yaml").write_text(yaml.safe_dump({"random_state": 1}))
        config = load_config(
            [tmp_path / "a.yaml"], overrides=["random_state=7"], project_root=tmp_path
        )
        assert config.random_state == 7

    def test_resolves_paths_against_the_project_root(self, tmp_path):
        (tmp_path / "a.yaml").write_text(yaml.safe_dump({}))
        config = load_config([tmp_path / "a.yaml"], project_root=tmp_path)
        assert config.paths.raw_dir.is_absolute()
        assert config.paths.raw_dir.is_relative_to(tmp_path)

    def test_reports_a_missing_file_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config([tmp_path / "nope.yaml"], project_root=tmp_path)

    def test_rejects_a_non_mapping_document(self, tmp_path):
        (tmp_path / "a.yaml").write_text("- just\n- a list\n")
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_config([tmp_path / "a.yaml"], project_root=tmp_path)

    def test_accepts_an_empty_document(self, tmp_path):
        (tmp_path / "a.yaml").write_text("")
        assert load_config([tmp_path / "a.yaml"], project_root=tmp_path).random_state == 42


class TestShippedConfigs:
    """The files in configs/ must actually load. They are executable, not prose."""

    def test_base_config_loads(self):
        config = load_config([REPO_CONFIGS / "base.yaml"])
        assert config.split.strategy == "out_of_time"
        assert config.evaluation.primary_metric == "average_precision"

    def test_base_plus_training_loads_and_enables_models(self):
        config = load_config([REPO_CONFIGS / "base.yaml", REPO_CONFIGS / "training.yaml"])
        training = config.require_training()
        assert len(training.enabled_models) >= 3

    def test_training_config_includes_a_no_skill_baseline(self):
        """Every comparison needs a floor. Without one, a mediocre model looks good."""
        config = load_config([REPO_CONFIGS / "base.yaml", REPO_CONFIGS / "training.yaml"])
        assert any(spec.estimator == "dummy" for spec in config.require_training().enabled_models)

    def test_base_plus_inference_loads(self):
        config = load_config([REPO_CONFIGS / "base.yaml", REPO_CONFIGS / "inference.yaml"])
        assert config.inference.validate_input is True

    def test_base_config_drops_the_post_call_column(self):
        config = load_config([REPO_CONFIGS / "base.yaml"])
        assert "duration" in config.data.drop_columns
