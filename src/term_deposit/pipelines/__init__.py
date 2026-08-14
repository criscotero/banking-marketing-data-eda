"""End-to-end orchestration: the only place stages are wired together."""

from __future__ import annotations

from term_deposit.pipelines.experiment import (
    ExperimentResult,
    PreparedData,
    prepare_dataset,
    run_experiment,
)

__all__ = ["ExperimentResult", "PreparedData", "prepare_dataset", "run_experiment"]
