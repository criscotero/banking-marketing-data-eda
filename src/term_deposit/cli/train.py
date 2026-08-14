"""Train and compare every configured model, then persist the winner.

    uv run python scripts/train.py

Runs the full experiment: split, fit, calibrate on validation, select a threshold
on validation, evaluate on test, cross-validate, backtest, and save the best model
with the metadata needed to reproduce it.

Compare protocols directly:

    uv run python scripts/train.py --set split.strategy=random
    uv run python scripts/train.py --set split.strategy=out_of_time
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from term_deposit.cli._shared import fail, resolve_config
from term_deposit.config import ConfigError
from term_deposit.data.loader import DatasetNotFoundError
from term_deposit.pipelines.experiment import run_experiment
from term_deposit.utils.logging import get_logger

app = typer.Typer(help=__doc__, no_args_is_help=False, add_completion=False)
logger = get_logger("cli.train")

_DISPLAY_COLUMNS = [
    "model",
    "average_precision",
    "roc_auc",
    "within_period_roc_auc",
    "lift_at_0.20",
    "precision_at_0.20",
    "brier_score",
    "backtest_ap_mean",
    "backtest_lift20_mean",
]


@app.command()
def main(
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="Config files, merged left to right."),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override a config key: dotted.key=value."),
    ] = None,
    no_persist: Annotated[
        bool, typer.Option("--no-persist", help="Skip writing artifacts and reports.")
    ] = False,
    no_download: Annotated[
        bool, typer.Option("--no-download", help="Fail if the dataset is absent.")
    ] = False,
    quick: Annotated[
        bool,
        typer.Option(
            "--quick",
            help="Skip cross-validation and the backtest. For smoke tests, not for reporting.",
        ),
    ] = False,
) -> None:
    """Run the training experiment."""
    overrides = list(set_ or [])
    if quick:
        overrides += ["training.cross_validate=false", "training.backtest=false"]

    try:
        app_config = resolve_config(
            config, overrides, extra_defaults=(Path("configs/training.yaml"),)
        )
        app_config.require_training()
    except ConfigError as error:
        fail(str(error))
        return

    try:
        result = run_experiment(app_config, persist=not no_persist, download=not no_download)
    except DatasetNotFoundError as error:
        fail(str(error))
        return

    columns = [column for column in _DISPLAY_COLUMNS if column in result.comparison.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        typer.echo("")
        typer.echo(f"Run {result.run_id}  (protocol: {result.split.strategy})")
        typer.echo(f"  split sizes         {result.split.sizes}")
        typer.echo(
            "  base rates          "
            + ", ".join(f"{k}={v:.4f}" for k, v in result.split.positive_rates.items())
        )
        if result.split.boundaries:
            typer.echo(f"  calendar boundaries {result.split.boundaries}")
        typer.echo("")
        typer.echo(result.comparison[columns].round(4).to_string(index=False))
        typer.echo("")
        typer.echo(f"Best model: {result.best.model_name}")
        typer.echo(
            f"  threshold {result.best.threshold.threshold:.4f} "
            f"(objective={result.best.threshold.objective}, "
            f"chosen on {result.best.threshold.chosen_on})"
        )
        if result.artifact_dir:
            typer.echo(f"  artifact  {result.artifact_dir}")

    inflation = result.best.within_period.get("roc_auc_inflation")
    if inflation is not None and inflation == inflation and inflation > 0.05:
        typer.echo("")
        typer.echo(
            f"note: pooled ROC-AUC exceeds the within-period value by {inflation:.3f}. "
            "That gap is calendar signal, not customer signal — quote the "
            "within-period number when describing campaign performance."
        )


if __name__ == "__main__":
    app()
