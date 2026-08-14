"""Typed, layered configuration.

Configuration is data, not code: every knob that changes a result lives in
``configs/*.yaml`` and is parsed into frozen pydantic models. Layers are merged
in order (``base.yaml`` first, then the task file, then explicit overrides), so a
training run and an inference run share one definition of paths and columns.

Nothing in this module reads the filesystem beyond the YAML files it is handed —
resolution of *which* files to read belongs to the callers in ``scripts/``.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from term_deposit import constants

DEFAULT_CONFIG_DIR = Path("configs")

SplitStrategy = Literal["random", "out_of_time"]
FeatureSet = Literal["all", "client_only"]


class _Frozen(BaseModel):
    """Base for every config node: immutable and intolerant of stray keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Frozen):
    """Filesystem layout, relative to the repository root unless absolute."""

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    @property
    def raw_csv(self) -> Path:
        """Location of the canonical raw CSV."""
        return self.raw_dir / constants.RAW_FILENAME

    @property
    def figures_dir(self) -> Path:
        """Directory for generated plots."""
        return self.reports_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        """Directory for generated metric documents."""
        return self.reports_dir / "metrics"

    def resolve(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path anchored at ``root``."""
        return PathsConfig(
            **{
                name: (value if value.is_absolute() else root / value)
                for name, value in self.model_dump().items()
            }
        )

    def ensure_directories(self) -> None:
        """Create every configured directory, including derived report subdirs."""
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.figures_dir,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class DataConfig(_Frozen):
    """How the raw dataset is obtained and validated."""

    source_url: str = constants.UCI_ARCHIVE_URL
    expected_sha256: str = constants.RAW_SHA256
    #: When false, a checksum mismatch logs a warning instead of raising. Useful
    #: for experimenting against a modified extract; never for a reported run.
    enforce_checksum: bool = True
    #: Fail fast if the raw file violates the declared schema.
    validate_schema: bool = True
    #: Columns dropped before modelling because they are unavailable pre-call.
    drop_columns: tuple[str, ...] = constants.POST_OUTCOME_COLUMNS


class SplitConfig(_Frozen):
    """Train/validation/test strategy.

    ``random`` reproduces the stratified shuffle used by the original notebook.
    ``out_of_time`` holds out the most recent calendar periods, which is the
    protocol that matches how the model would actually be deployed (ADR 0002).
    """

    strategy: SplitStrategy = "out_of_time"
    test_size: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2
    validation_size: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.15
    #: Number of trailing calendar months reserved for the out-of-time test set.
    test_periods: Annotated[int, Field(ge=1)] = 9
    #: Number of calendar months before the test window used for validation.
    validation_periods: Annotated[int, Field(ge=1)] = 7
    random_state: int = 42

    @model_validator(mode="after")
    def _check_random_fractions(self) -> SplitConfig:
        if self.strategy == "random" and self.test_size + self.validation_size >= 1.0:
            msg = "test_size + validation_size must leave room for a training set"
            raise ValueError(msg)
        return self


class FeatureConfig(_Frozen):
    """Which columns reach the model and how they are transformed."""

    feature_set: FeatureSet = "all"
    #: Split ``pdays`` into a "never contacted" flag plus a real elapsed-days
    #: column, instead of scaling the 999 sentinel as if it were a duration.
    encode_pdays_sentinel: bool = True
    #: Scale numeric columns. Required for the linear baseline; harmless for trees.
    scale_numeric: bool = True
    #: Unseen categories at inference time become an all-zero block rather than
    #: raising, so a new job title cannot take the scorer down.
    handle_unknown_categories: Literal["ignore", "error"] = "ignore"

    def numeric_columns(self) -> tuple[str, ...]:
        """Numeric input columns implied by ``feature_set``."""
        if self.feature_set == "client_only":
            return (*constants.CLIENT_NUMERIC_FEATURES, *constants.CAMPAIGN_NUMERIC_FEATURES)
        return constants.NUMERIC_FEATURES

    def categorical_columns(self) -> tuple[str, ...]:
        """Categorical input columns implied by ``feature_set``."""
        return constants.CATEGORICAL_FEATURES

    def input_columns(self) -> tuple[str, ...]:
        """Every raw column the preprocessor expects, in a stable order."""
        return (*self.numeric_columns(), *self.categorical_columns())


class ModelSpec(_Frozen):
    """One candidate estimator.

    ``params`` is passed verbatim to the estimator constructor, so hyperparameters
    stay in YAML rather than being buried in Python. ``balance_strategy`` records
    *how* imbalance is handled, which the trainer needs in order to fill in
    weights that depend on the training fold (``scale_pos_weight``).
    """

    name: str
    estimator: Literal[
        "dummy",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ]
    params: Mapping[str, Any] = Field(default_factory=dict)
    balance_strategy: Literal["none", "class_weight", "scale_pos_weight"] = "none"
    enabled: bool = True


class CalibrationConfig(_Frozen):
    """Post-hoc probability calibration.

    Class-weighted estimators emit scores that rank well but are systematically
    inflated. Calibration is fitted on the validation split only.
    """

    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "isotonic"


class TrainingConfig(_Frozen):
    """Everything the training entry point needs."""

    models: tuple[ModelSpec, ...]
    calibration: CalibrationConfig = CalibrationConfig()
    #: Folds for the cross-validated sanity check on the training split.
    cv_folds: Annotated[int, Field(ge=2)] = 5
    #: Run the (slower) cross-validation stage.
    cross_validate: bool = True
    #: Also run a rolling-origin backtest, one fold per trailing calendar month.
    backtest: bool = True
    backtest_periods: Annotated[int, Field(ge=1)] = 9
    #: Minimum rows in a backtest fold before it is scored.
    backtest_min_rows: Annotated[int, Field(ge=1)] = 100
    random_state: int = 42
    n_jobs: int = -1

    @property
    def enabled_models(self) -> tuple[ModelSpec, ...]:
        """Model specs with ``enabled: true``."""
        return tuple(spec for spec in self.models if spec.enabled)

    @model_validator(mode="after")
    def _require_unique_enabled_models(self) -> TrainingConfig:
        names = [spec.name for spec in self.models]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"duplicate model names in training config: {duplicates}"
            raise ValueError(msg)
        if not self.enabled_models:
            msg = "training config enables no models"
            raise ValueError(msg)
        return self


class EvaluationConfig(_Frozen):
    """Which numbers are computed and which one decides the winner."""

    #: Primary metric. Average precision, not ROC-AUC: with an 11% base rate the
    #: ROC curve is dominated by the majority class. See ADR 0005.
    primary_metric: str = "average_precision"
    #: Capacity fractions for lift/precision@k — the call-centre framing.
    top_k_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    #: Rule used to pick the decision threshold on the *validation* split.
    threshold_objective: Literal["f1", "expected_value", "top_k"] = "expected_value"
    #: Fraction of the list the campaign can actually call, for ``top_k``.
    capacity_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    #: Contribution of one subscription, in arbitrary but consistent units.
    value_per_subscription: float = 100.0
    #: Cost of one outbound call attempt, same units.
    cost_per_call: float = 5.0
    #: Report ranking quality *within* each calendar month, which is the only
    #: comparison a live campaign ever makes (ADR 0003).
    report_within_period: bool = True
    #: Periods smaller than this are skipped when averaging within-period scores.
    within_period_min_rows: Annotated[int, Field(ge=2)] = 50
    n_bootstrap: Annotated[int, Field(ge=0)] = 0


class ExplainabilityConfig(_Frozen):
    """SHAP settings. Optional: requires the ``explain`` extra."""

    enabled: bool = False
    background_size: Annotated[int, Field(ge=1)] = 200
    max_display: Annotated[int, Field(ge=1)] = 20
    #: Cap on rows explained; SHAP over the full test set is slow and adds nothing.
    sample_size: Annotated[int, Field(ge=1)] = 2000
    random_state: int = 42


class TrackingConfig(_Frozen):
    """Experiment tracking backend.

    ``jsonl`` needs no service and no extra dependency, which keeps ``uv sync``
    honest for anyone cloning the repo. ``mlflow`` is available behind an extra
    for anyone who wants the UI. See ADR 0006.
    """

    backend: Literal["jsonl", "mlflow", "none"] = "jsonl"
    experiment_name: str = "term-deposit-propensity"
    #: MLflow tracking URI; ignored by the other backends.
    tracking_uri: str | None = None


class InferenceConfig(_Frozen):
    """Batch scoring behaviour."""

    #: Which artifact to score with. ``latest`` resolves the newest run directory.
    model_id: str = "latest"
    batch_size: Annotated[int, Field(ge=1)] = 10_000
    #: Emit the tier label alongside the probability.
    include_tier: bool = True
    #: Upper probability bound of each tier, from best to worst.
    tier_quantiles: tuple[float, ...] = (0.05, 0.20, 0.50)
    tier_labels: tuple[str, ...] = ("tier_1", "tier_2", "tier_3", "tier_4")
    #: Refuse to score if the input violates the declared schema.
    validate_input: bool = True

    @model_validator(mode="after")
    def _check_tiers(self) -> InferenceConfig:
        if len(self.tier_labels) != len(self.tier_quantiles) + 1:
            msg = (
                f"tier_labels must have one more entry than tier_quantiles "
                f"({len(self.tier_labels)} vs {len(self.tier_quantiles)} + 1)"
            )
            raise ValueError(msg)
        if list(self.tier_quantiles) != sorted(self.tier_quantiles):
            msg = "tier_quantiles must be ascending"
            raise ValueError(msg)
        return self


class AppConfig(_Frozen):
    """Root configuration object handed to every entry point."""

    project_name: str = "term-deposit-propensity"
    #: Seeds numpy, python-random and every estimator that accepts a seed.
    random_state: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    features: FeatureConfig = FeatureConfig()
    training: TrainingConfig | None = None
    evaluation: EvaluationConfig = EvaluationConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    tracking: TrackingConfig = TrackingConfig()
    inference: InferenceConfig = InferenceConfig()

    def require_training(self) -> TrainingConfig:
        """Return the training config or fail with an actionable message."""
        if self.training is None:
            msg = (
                "no training section is configured; load configs/training.yaml "
                "(e.g. `--config configs/base.yaml --config configs/training.yaml`)"
            )
            raise ConfigError(msg)
        return self.training


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed or contradictory."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Mappings merge key-by-key; every other type (including lists) is replaced
    wholesale, so a config file can shorten a model list rather than only extend it.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning ``{}`` for an empty document."""
    if not path.is_file():
        msg = f"config file not found: {path}"
        raise ConfigError(msg)
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"config file {path} must contain a mapping at the top level"
        raise ConfigError(msg)
    return loaded


def parse_override(raw: str) -> tuple[Sequence[str], Any]:
    """Parse a ``dotted.key=value`` CLI override into a path and a typed value.

    The value is parsed as YAML, so ``split.strategy=random``, ``training.cv_folds=3``
    and ``evaluation.top_k_fractions=[0.1, 0.2]`` all behave as expected.
    """
    key, separator, value = raw.partition("=")
    if not separator:
        msg = f"override {raw!r} is not of the form key=value"
        raise ConfigError(msg)
    key = key.strip()
    if not key:
        msg = f"override {raw!r} has an empty key"
        raise ConfigError(msg)
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as error:  # pragma: no cover - defensive
        msg = f"could not parse override value in {raw!r}: {error}"
        raise ConfigError(msg) from error
    return key.split("."), parsed


def apply_overrides(payload: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides on top of a config payload."""
    result: dict[str, Any] = copy.deepcopy(dict(payload))
    for raw in overrides:
        path, value = parse_override(raw)
        cursor = result
        for part in path[:-1]:
            nested = cursor.get(part)
            if not isinstance(nested, dict):
                nested = {}
                cursor[part] = nested
            cursor = nested
        cursor[path[-1]] = value
    return result


def load_config(
    paths: Sequence[Path] | None = None,
    *,
    overrides: Iterable[str] = (),
    project_root: Path | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from layered YAML files and CLI overrides.

    Args:
        paths: Config files, merged left to right. Defaults to ``configs/base.yaml``.
        overrides: ``dotted.key=value`` strings applied after the files.
        project_root: Anchor for relative paths. Defaults to ``TERM_DEPOSIT_ROOT``
            or the repository root inferred from this file's location.

    Returns:
        A frozen configuration with all paths resolved to absolute locations.

    Raises:
        ConfigError: If a file is missing, malformed, or the merged result fails
            validation.
    """
    files = tuple(paths) if paths else (DEFAULT_CONFIG_DIR / "base.yaml",)
    payload: dict[str, Any] = {}
    for path in files:
        payload = deep_merge(payload, load_yaml(path))
    payload = apply_overrides(payload, overrides)

    try:
        config = AppConfig.model_validate(payload)
    except Exception as error:  # pydantic raises ValidationError
        msg = f"invalid configuration from {[str(p) for p in files]}: {error}"
        raise ConfigError(msg) from error

    root = project_root or _default_project_root()
    return config.model_copy(update={"paths": config.paths.resolve(root)})


def _default_project_root() -> Path:
    """Repository root: ``$TERM_DEPOSIT_ROOT`` if set, else three levels up."""
    env_root = os.environ.get("TERM_DEPOSIT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # src/term_deposit/config.py -> src/term_deposit -> src -> <root>
    return Path(__file__).resolve().parents[2]
