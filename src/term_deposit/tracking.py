"""Experiment tracking behind a narrow interface.

The pipeline depends on the :class:`ExperimentTracker` protocol, not on a tracking
product. The default :class:`JsonlTracker` writes append-only records to
``reports/metrics/`` — no service, no extra dependency, and the history is
diffable in git. :class:`MlflowTracker` is available for anyone who wants the UI
and is installed only via the ``mlflow`` extra.

Making the heavyweight option opt-in keeps `uv sync` fast and keeps a fresh clone
runnable offline, which matters more for a reviewed portfolio project than a
tracking server does. See ADR 0006.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from term_deposit.config import TrackingConfig
from term_deposit.utils.logging import get_logger
from term_deposit.utils.serialization import append_jsonl, write_json

logger = get_logger(__name__)


@runtime_checkable
class ExperimentTracker(Protocol):
    """What the training pipeline needs from a tracking backend."""

    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> AbstractContextManager[None]:
        """Open a run scope. The scope yields nothing; callers use the tracker."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record the inputs of a run."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record scalar outputs."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Attach a file produced by the run."""
        ...


class _RunScope(AbstractContextManager[None]):
    """Context manager returned by trackers, flushing the run on exit.

    It yields nothing: callers log through the tracker they already hold, so
    there is no reason for ``with`` to rebind a name.
    """

    def __init__(self, tracker: NullTracker, run_name: str) -> None:
        """Bind the scope to a tracker and a run name."""
        self._tracker = tracker
        self._run_name = run_name

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._tracker.end_run(self._run_name, failed=exc is not None)


class NullTracker:
    """Records nothing. Used by tests and by ``tracking.backend: none``."""

    def __init__(self) -> None:
        """Start with empty in-memory params, metrics and artifact lists."""
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[Path] = []

    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> AbstractContextManager[None]:
        """Open a no-op run scope."""
        del tags
        return _RunScope(self, run_name)

    def end_run(self, run_name: str, *, failed: bool = False) -> None:
        """Close the scope."""
        del run_name, failed

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Store parameters in memory."""
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Store metrics in memory."""
        del step
        self.metrics.update(metrics)

    def log_artifact(self, path: Path) -> None:
        """Remember an artifact path."""
        self.artifacts.append(path)


class JsonlTracker(NullTracker):
    """Append-only, file-based tracking.

    Every run appends one JSON line to ``runs.jsonl`` and writes a full record to
    ``runs/<run_name>.json``. That is enough to compare experiments with pandas
    and enough to review a run's provenance in a pull request.
    """

    def __init__(self, output_dir: Path, experiment_name: str) -> None:
        """Write run records under ``output_dir`` for ``experiment_name``."""
        super().__init__()
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.index_path = output_dir / "runs.jsonl"
        self._tags: dict[str, str] = {}

    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> AbstractContextManager[None]:
        """Begin a run, clearing any state from the previous one."""
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self._tags = dict(tags or {})
        return _RunScope(self, run_name)

    def end_run(self, run_name: str, *, failed: bool = False) -> None:
        """Flush the run to ``runs.jsonl`` and to its own JSON document."""
        record = {
            "experiment": self.experiment_name,
            "run_name": run_name,
            "status": "failed" if failed else "finished",
            "tags": self._tags,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": [str(path) for path in self.artifacts],
        }
        append_jsonl(self.index_path, record)
        write_json(self.output_dir / "runs" / f"{run_name}.json", record)
        logger.info("Tracked run %r -> %s", run_name, self.index_path)


class MlflowTracker:
    """MLflow-backed tracking. Requires the ``mlflow`` extra."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        """Connect to MLflow.

        Raises:
            ImportError: If the ``mlflow`` extra is not installed.
        """
        try:
            import mlflow
        except ImportError as error:  # pragma: no cover - optional dependency
            msg = (
                "tracking.backend is 'mlflow' but mlflow is not installed. "
                "Install it with `uv sync --extra mlflow`, or set tracking.backend "
                "to 'jsonl'."
            )
            raise ImportError(msg) from error
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> AbstractContextManager[None]:
        """Open an MLflow run."""
        scope: AbstractContextManager[None] = self._mlflow.start_run(
            run_name=run_name, tags=dict(tags or {})
        )
        return scope

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Forward parameters to MLflow, stringifying nested values."""
        self._mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Forward finite scalar metrics to MLflow."""
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and float(value) == float(value)
        }
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path) -> None:
        """Upload a file or directory to the run."""
        if path.is_dir():
            self._mlflow.log_artifacts(str(path))
        else:
            self._mlflow.log_artifact(str(path))


def build_tracker(config: TrackingConfig, output_dir: Path) -> ExperimentTracker:
    """Construct the tracker named by configuration.

    Args:
        config: Tracking settings.
        output_dir: Where the JSONL backend writes its records.

    Returns:
        A tracker satisfying :class:`ExperimentTracker`.
    """
    if config.backend == "none":
        return NullTracker()
    if config.backend == "mlflow":
        return MlflowTracker(config.experiment_name, config.tracking_uri)
    return JsonlTracker(output_dir, config.experiment_name)
