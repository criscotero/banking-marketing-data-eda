"""The training experiment, start to finish.

    load -> validate -> reconstruct calendar -> split -> train -> calibrate
         -> select threshold -> evaluate -> persist -> track

Each step is a call into a module that can be tested on its own; this file owns
the *order* and nothing else. Adding a stage means editing one function, and the
scripts stay thin wrappers over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from term_deposit import constants
from term_deposit.config import AppConfig
from term_deposit.data.loader import ensure_raw_dataset, load_raw_dataset, sha256_of_file
from term_deposit.data.schema import summarise_quality
from term_deposit.data.splits import DataSplit, describe_drift, make_split
from term_deposit.evaluation.report import (
    EvaluationReport,
    build_comparison_table,
    evaluate_all,
    select_best,
    write_reports,
)
from term_deposit.features.calendar import (
    add_contact_period,
    macro_period_collinearity,
    period_summary,
)
from term_deposit.models.persistence import ModelArtifact, ModelMetadata, save_artifact
from term_deposit.tracking import build_tracker
from term_deposit.training.trainer import TrainedModel, train_all
from term_deposit.utils.logging import get_logger
from term_deposit.utils.seeding import seed_everything
from term_deposit.utils.serialization import write_json

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedData:
    """The dataset after loading, validation and calendar reconstruction."""

    frame: pd.DataFrame
    checksum: str | None
    quality: pd.DataFrame
    periods: pd.DataFrame
    macro_collinearity: pd.DataFrame

    @property
    def labels(self) -> pd.Series:
        """Binary target."""
        return self.frame[constants.LABEL_COLUMN]

    @property
    def period_key(self) -> pd.Series:
        """Reconstructed calendar period per row."""
        return self.frame[constants.PERIOD_COLUMN]

    def features(self, columns: tuple[str, ...]) -> pd.DataFrame:
        """Feature frame restricted to ``columns``."""
        return self.frame.loc[:, list(columns)]


def prepare_dataset(config: AppConfig, *, download: bool = True) -> PreparedData:
    """Load the raw CSV, validate it and attach the reconstructed calendar.

    ``duration`` is dropped here rather than at model-fitting time, so no later
    stage can reintroduce it by accident.

    Args:
        config: Resolved application configuration.
        download: Fetch the dataset if it is missing.

    Returns:
        A :class:`PreparedData` bundle.
    """
    raw_path = ensure_raw_dataset(config.data, config.paths.raw_csv, download=download)
    checksum = sha256_of_file(raw_path)

    frame = load_raw_dataset(raw_path, validate=config.data.validate_schema)
    quality = summarise_quality(frame)

    dropped = [column for column in config.data.drop_columns if column in frame.columns]
    if dropped:
        logger.info("Dropping post-outcome column(s) to prevent target leakage: %s", dropped)
        frame = frame.drop(columns=dropped)

    frame = add_contact_period(frame)
    periods = period_summary(frame)
    collinearity = macro_period_collinearity(frame)

    logger.info(
        "Prepared %d rows across %d calendar periods (%s..%s); base rate %.4f",
        len(frame),
        frame[constants.PERIOD_COLUMN].nunique(),
        frame[constants.PERIOD_COLUMN].min(),
        frame[constants.PERIOD_COLUMN].max(),
        float(frame[constants.LABEL_COLUMN].mean()),
    )
    return PreparedData(
        frame=frame,
        checksum=checksum,
        quality=quality,
        periods=periods,
        macro_collinearity=collinearity,
    )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything one training run produced."""

    run_id: str
    split: DataSplit
    models: list[TrainedModel]
    reports: list[EvaluationReport]
    comparison: pd.DataFrame
    best: EvaluationReport
    artifact_dir: Path | None
    written: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Serialisable run summary, written to ``reports/metrics/<run_id>.json``."""
        return {
            "run_id": self.run_id,
            "split": self.split.summary(),
            "best_model": self.best.model_name,
            "comparison": self.comparison.to_dict(orient="records"),
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
        }


def _run_id(config: AppConfig, timestamp: datetime | None = None) -> str:
    """Human-sortable run identifier: timestamp, protocol and feature set.

    Second resolution keeps the identifier readable, so two runs started inside
    the same second would collide and the later one would overwrite the earlier
    one's artifact directory. A numeric suffix is appended when that happens
    rather than losing a run.
    """
    stamp = (timestamp or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}__{config.split.strategy}__{config.features.feature_set}"

    artifacts_dir = config.paths.artifacts_dir
    if not (artifacts_dir / base).exists():
        return base
    suffix = 2
    while (artifacts_dir / f"{base}-{suffix}").exists():
        suffix += 1
    logger.debug("Run id %s already exists; using %s-%d", base, base, suffix)
    return f"{base}-{suffix}"


def run_experiment(
    config: AppConfig,
    *,
    data: PreparedData | None = None,
    persist: bool = True,
    download: bool = True,
    timestamp: datetime | None = None,
) -> ExperimentResult:
    """Train, evaluate, persist and track one experiment.

    Args:
        config: Resolved application configuration, including a training section.
        data: Pre-loaded dataset. Loaded from disk when omitted.
        persist: Write the winning artifact and the metric reports.
        download: Allow the loader to fetch the dataset if it is missing.
        timestamp: Override the run timestamp, for deterministic tests.

    Returns:
        An :class:`ExperimentResult`.

    Raises:
        ConfigError: If no training section is configured.
    """
    training = config.require_training()
    seed_everything(config.random_state)
    config.paths.ensure_directories()

    prepared = data if data is not None else prepare_dataset(config, download=download)
    feature_columns = config.features.input_columns()

    split = make_split(prepared.frame, config.split, feature_columns=feature_columns)

    run_id = _run_id(config, timestamp)
    tracker = build_tracker(config.tracking, config.paths.metrics_dir)

    with tracker.start_run(run_id, tags={"split": config.split.strategy}):
        tracker.log_params(
            {
                "split_strategy": config.split.strategy,
                "feature_set": config.features.feature_set,
                "models": [spec.name for spec in training.enabled_models],
                "random_state": config.random_state,
                "calibration": training.calibration.method
                if training.calibration.enabled
                else "none",
                "threshold_objective": config.evaluation.threshold_objective,
                "data_checksum": prepared.checksum,
                **{f"n_{name}": size for name, size in split.sizes.items()},
            }
        )

        models = train_all(
            split,
            config,
            frame=prepared.features(feature_columns),
            labels=prepared.labels,
            periods=prepared.period_key,
        )
        reports = evaluate_all(models, split, config)
        comparison = build_comparison_table(
            reports, primary_metric=config.evaluation.primary_metric
        )
        best = select_best(
            reports,
            primary_metric=config.evaluation.primary_metric,
            prefer_backtest=training.backtest,
        )

        tracker.log_metrics(
            {
                f"{report.model_name}::test_{config.evaluation.primary_metric}": getattr(
                    report.test_metrics, config.evaluation.primary_metric
                )
                for report in reports
            }
        )
        tracker.log_metrics(
            {
                f"{report.model_name}::within_period_roc_auc": report.within_period.get(
                    "weighted_roc_auc", float("nan")
                )
                for report in reports
            }
        )

        written: dict[str, Path] = {}
        artifact_dir: Path | None = None

        if persist:
            written = write_reports(
                reports,
                config.paths.metrics_dir,
                primary_metric=config.evaluation.primary_metric,
            )
            written["run_summary"] = write_json(
                config.paths.metrics_dir / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "config": config.model_dump(mode="json"),
                    "data": {
                        "checksum": prepared.checksum,
                        "n_rows": len(prepared.frame),
                        "period_summary": prepared.periods.assign(
                            contact_period=lambda d: d["contact_period"].astype(str)
                        ).to_dict(orient="records"),
                        "macro_period_collinearity": prepared.macro_collinearity.to_dict(
                            orient="records"
                        ),
                    },
                    "split": split.summary(),
                    "drift": describe_drift(split).head(15).to_dict(orient="records"),
                    "comparison": comparison.to_dict(orient="records"),
                    "best_model": best.model_name,
                },
            )
            artifact_dir = _persist_best(config, run_id, models, best, split, prepared)
            tracker.log_artifact(artifact_dir)

    return ExperimentResult(
        run_id=run_id,
        split=split,
        models=models,
        reports=reports,
        comparison=comparison,
        best=best,
        artifact_dir=artifact_dir,
        written=written,
    )


def _persist_best(
    config: AppConfig,
    run_id: str,
    models: list[TrainedModel],
    best: EvaluationReport,
    split: DataSplit,
    prepared: PreparedData,
) -> Path:
    """Save the winning model together with the metadata needed to trust it."""
    winner = next(model for model in models if model.name == best.model_name)

    notes = [
        "Metrics were produced under the "
        f"'{split.strategy}' protocol; see docs/methodology.md for what that means.",
        "The macro indicators are near-constant within a calendar month, so pooled "
        "ranking metrics overstate within-campaign performance. Compare "
        "within_period_roc_auc against roc_auc before quoting a number.",
        "'duration' is excluded: it is only known after a call ends.",
    ]
    if winner.calibrator is None:
        notes.append("Probabilities are uncalibrated; treat them as scores, not as rates.")

    metadata = ModelMetadata.create(
        model_name=winner.name,
        estimator=winner.spec.estimator,
        split_strategy=split.strategy,
        feature_set=config.features.feature_set,
        input_columns=config.features.input_columns(),
        decision_threshold=best.threshold.threshold,
        threshold_objective=best.threshold.objective,
        metrics={
            "test": best.test_metrics.to_dict(),
            "within_period": {
                key: value for key, value in best.within_period.items() if key != "by_period"
            },
            "backtest": (best.backtest or {}).get("summary", {}),
            "cross_validation": best.cross_validation,
        },
        config=config.model_dump(mode="json"),
        data_checksum=prepared.checksum,
        notes=tuple(notes),
    )

    # The calibrated scorer is what produced the reported metrics, so it is what
    # gets shipped. Saving the raw pipeline instead would mean the artifact and
    # the report disagreed.
    shipped = winner.scorer if winner.calibrator is not None else winner.pipeline
    artifact = ModelArtifact(pipeline=shipped, metadata=metadata)
    return save_artifact(artifact, config.paths.artifacts_dir / run_id)
