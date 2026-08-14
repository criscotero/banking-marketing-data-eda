"""Model artifacts: what gets saved, and what has to travel with it.

A pickled estimator on its own is not reproducible. An artifact here is the fitted
pipeline *plus* the metadata needed to answer, months later: which data, which
config, which library versions, which threshold, and what did it score?

Preprocessing is inside the pipeline, so the artifact is self-contained — the
inference path cannot drift from the training path because there is only one
transformation and it is in the file.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from term_deposit import __version__
from term_deposit.utils.logging import get_logger
from term_deposit.utils.serialization import read_json, write_json

logger = get_logger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
LATEST_LINK = "latest"


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be written, found or read back."""


def _git_revision() -> str | None:
    """Short commit hash of the working tree, or ``None`` outside a repo."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _library_versions() -> dict[str, str]:
    """Versions of the libraries that can change a prediction."""
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "term_deposit": __version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:  # pragma: no cover - optional at runtime
        pass
    try:
        import pandas

        versions["pandas"] = pandas.__version__
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        pass
    return versions


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Everything needed to interpret, audit or reproduce a saved model.

    Attributes:
        model_name: Name from the training config.
        estimator: Registry key of the underlying estimator.
        split_strategy: Which evaluation protocol produced the reported metrics.
        feature_set: ``all`` or ``client_only``.
        input_columns: Raw columns the pipeline expects, in order. The inference
            schema is derived from this, so a renamed column fails loudly.
        decision_threshold: Probability cut-off chosen on the *validation* split.
        threshold_objective: Rule used to choose it.
        metrics: Test-set metrics as reported.
        created_at: UTC ISO-8601 timestamp.
        git_revision: Commit the run was produced from, when available.
        library_versions: Versions of prediction-affecting libraries.
        config: The resolved configuration used for the run.
        data_checksum: SHA-256 of the raw dataset.
        notes: Free-form caveats surfaced in the model card.
    """

    model_name: str
    estimator: str
    split_strategy: str
    feature_set: str
    input_columns: tuple[str, ...]
    decision_threshold: float
    threshold_objective: str
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    git_revision: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    data_checksum: str | None = None
    notes: tuple[str, ...] = ()
    schema_version: int = 1

    @classmethod
    def create(cls, **kwargs: Any) -> ModelMetadata:
        """Build metadata, stamping the environment automatically."""
        kwargs.setdefault("created_at", datetime.now(UTC).isoformat())
        kwargs.setdefault("git_revision", _git_revision())
        kwargs.setdefault("library_versions", _library_versions())
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "estimator": self.estimator,
            "split_strategy": self.split_strategy,
            "feature_set": self.feature_set,
            "input_columns": list(self.input_columns),
            "decision_threshold": self.decision_threshold,
            "threshold_objective": self.threshold_objective,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "git_revision": self.git_revision,
            "library_versions": self.library_versions,
            "config": self.config,
            "data_checksum": self.data_checksum,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelMetadata:
        """Rebuild metadata from its JSON form."""
        known = {
            "model_name",
            "estimator",
            "split_strategy",
            "feature_set",
            "decision_threshold",
            "threshold_objective",
            "metrics",
            "created_at",
            "git_revision",
            "library_versions",
            "config",
            "data_checksum",
            "schema_version",
        }
        kwargs: dict[str, Any] = {key: payload[key] for key in known if key in payload}
        kwargs["input_columns"] = tuple(payload.get("input_columns", ()))
        kwargs["notes"] = tuple(payload.get("notes", ()))
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A fitted pipeline together with its metadata."""

    pipeline: Pipeline
    metadata: ModelMetadata

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        """Positive-class probabilities for ``X``."""
        return np.asarray(self.pipeline.predict_proba(X))[:, 1]


def save_artifact(
    artifact: ModelArtifact,
    directory: Path,
    *,
    update_latest: bool = True,
) -> Path:
    """Persist a fitted pipeline and its metadata under ``directory``.

    Args:
        artifact: Fitted pipeline plus metadata.
        directory: Run directory. Created if absent.
        update_latest: Also refresh the sibling ``latest`` pointer, so that
            ``scripts/predict.py`` has a stable default target.

    Returns:
        The directory the artifact was written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    joblib.dump(artifact.pipeline, model_path, compress=3)
    write_json(directory / METADATA_FILENAME, artifact.metadata.to_dict())
    logger.info("Saved model artifact to %s", directory)

    if update_latest:
        _write_latest_pointer(directory)
    return directory


def _write_latest_pointer(directory: Path) -> None:
    """Point ``<parent>/latest`` at ``directory``.

    A symlink where the filesystem allows it, a text file otherwise, so the
    repository behaves the same on Windows and inside containers.
    """
    pointer = directory.parent / LATEST_LINK
    try:
        if pointer.is_symlink() or pointer.exists():
            if pointer.is_dir() and not pointer.is_symlink():
                (pointer / "TARGET").write_text(directory.name, encoding="utf-8")
                return
            pointer.unlink()
        pointer.symlink_to(directory.name, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pointer.with_suffix(".txt").write_text(directory.name, encoding="utf-8")


def resolve_artifact_path(artifacts_dir: Path, model_id: str = "latest") -> Path:
    """Resolve a model identifier to a concrete run directory.

    ``latest`` follows the pointer written by :func:`save_artifact`, falling back
    to the most recently modified run directory if the pointer is missing.

    Raises:
        ArtifactError: If nothing matching ``model_id`` can be found.
    """
    if model_id != LATEST_LINK:
        candidate = artifacts_dir / model_id
        if (candidate / MODEL_FILENAME).is_file():
            return candidate
        msg = f"no model artifact at {candidate}"
        raise ArtifactError(msg)

    pointer = artifacts_dir / LATEST_LINK
    if (pointer / MODEL_FILENAME).is_file():
        return pointer

    text_pointer = artifacts_dir / f"{LATEST_LINK}.txt"
    if text_pointer.is_file():
        target = artifacts_dir / text_pointer.read_text(encoding="utf-8").strip()
        if (target / MODEL_FILENAME).is_file():
            return target

    candidates = list_artifacts(artifacts_dir)
    if candidates:
        return candidates[-1]

    msg = (
        f"no model artifacts found under {artifacts_dir}. "
        "Run `uv run python scripts/train.py` first."
    )
    raise ArtifactError(msg)


def list_artifacts(artifacts_dir: Path) -> list[Path]:
    """Run directories containing a saved model, oldest modification first."""
    if not artifacts_dir.is_dir():
        return []
    found = [
        child
        for child in artifacts_dir.iterdir()
        if child.is_dir() and child.name != LATEST_LINK and (child / MODEL_FILENAME).is_file()
    ]
    return sorted(found, key=lambda path: path.stat().st_mtime)


def load_artifact(directory: Path) -> ModelArtifact:
    """Load a saved pipeline and its metadata.

    Raises:
        ArtifactError: If either file is missing or cannot be deserialised.
    """
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    if not model_path.is_file():
        msg = f"model file not found at {model_path}"
        raise ArtifactError(msg)
    if not metadata_path.is_file():
        msg = f"metadata file not found at {metadata_path}"
        raise ArtifactError(msg)

    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        msg = (
            f"could not deserialise {model_path}: {error}. "
            "Artifacts are version-sensitive; check metadata.json for the "
            "library versions the model was trained with."
        )
        raise ArtifactError(msg) from error

    metadata = ModelMetadata.from_dict(read_json(metadata_path))
    _warn_on_version_drift(metadata)
    return ModelArtifact(pipeline=pipeline, metadata=metadata)


def _warn_on_version_drift(metadata: ModelMetadata) -> None:
    """Log a warning when the loading environment differs from the training one."""
    current = _library_versions()
    drifted = {
        name: (recorded, current[name])
        for name, recorded in metadata.library_versions.items()
        if name in current and current[name] != recorded
    }
    if drifted:
        logger.warning(
            "Library versions differ from training time (%s); predictions may not "
            "match the recorded metrics",
            ", ".join(f"{n}: {old} -> {new}" for n, (old, new) in sorted(drifted.items())),
        )
