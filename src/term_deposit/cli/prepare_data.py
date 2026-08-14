"""Download, verify and profile the raw dataset.

    uv run python scripts/prepare_data.py

Downloads ``bank-additional-full.csv`` from UCI, checks its SHA-256, validates it
against the declared schema, reconstructs the contact calendar and writes the
data-quality and per-period profiles that the notebooks and README depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from term_deposit.cli._shared import fail, resolve_config
from term_deposit.config import ConfigError
from term_deposit.data.loader import DatasetNotFoundError, copy_local_dataset, write_sample
from term_deposit.pipelines.experiment import prepare_dataset
from term_deposit.utils.logging import get_logger
from term_deposit.utils.serialization import write_json

app = typer.Typer(help=__doc__, no_args_is_help=False, add_completion=False)
logger = get_logger("cli.prepare_data")


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
    from_file: Annotated[
        Path | None,
        typer.Option("--from-file", help="Use a local CSV instead of downloading."),
    ] = None,
    no_download: Annotated[
        bool,
        typer.Option(
            "--no-download", help="Fail if the dataset is absent rather than fetching it."
        ),
    ] = False,
    sample_rows: Annotated[
        int,
        typer.Option("--sample-rows", help="Rows to write to the interim sample. 0 disables."),
    ] = 2000,
) -> None:
    """Fetch and profile the dataset."""
    try:
        app_config = resolve_config(config, set_)
    except ConfigError as error:
        fail(str(error))
        return

    app_config.paths.ensure_directories()

    if from_file is not None:
        if not from_file.is_file():
            fail(f"--from-file does not exist: {from_file}")
            return
        copy_local_dataset(from_file, app_config.paths.raw_csv)

    try:
        prepared = prepare_dataset(app_config, download=not no_download)
    except DatasetNotFoundError as error:
        fail(str(error))
        return
    except Exception as error:  # schema or checksum failure
        fail(f"dataset preparation failed: {error}")
        return

    interim = app_config.paths.interim_dir
    prepared.quality.to_csv(interim / "data_quality.csv", index=False)
    periods = prepared.periods.assign(contact_period=lambda d: d["contact_period"].astype(str))
    periods.to_csv(interim / "period_summary.csv", index=False)
    prepared.macro_collinearity.to_csv(interim / "macro_period_collinearity.csv", index=False)

    if sample_rows > 0:
        write_sample(app_config.paths.raw_csv, interim / "sample.csv", n_rows=sample_rows)

    profile_path = write_json(
        app_config.paths.interim_dir / "dataset_profile.json",
        {
            "checksum_sha256": prepared.checksum,
            "n_rows": len(prepared.frame),
            "n_columns": prepared.frame.shape[1],
            "base_rate": float(prepared.labels.mean()),
            "n_periods": int(prepared.period_key.nunique()),
            "first_period": str(prepared.period_key.min()),
            "last_period": str(prepared.period_key.max()),
            "period_summary": periods.to_dict(orient="records"),
            "macro_period_collinearity": prepared.macro_collinearity.to_dict(orient="records"),
        },
    )

    typer.echo(f"Dataset ready:      {app_config.paths.raw_csv}")
    typer.echo(f"  rows             {len(prepared.frame):,}")
    typer.echo(f"  base rate        {prepared.labels.mean():.4f}")
    typer.echo(
        f"  periods          {prepared.period_key.nunique()} "
        f"({prepared.period_key.min()} .. {prepared.period_key.max()})"
    )
    typer.echo(f"  sha256           {prepared.checksum}")
    typer.echo(f"Profile written to  {profile_path}")

    share = prepared.macro_collinearity["between_period_variance_share"]
    if not share.empty:
        typer.echo(
            f"  macro features are {share.min():.1%}-{share.max():.1%} determined by the "
            "contact month (see docs/methodology.md)"
        )


if __name__ == "__main__":
    app()
