r"""Score a batch of customers and write a prioritised call list.

    uv run python scripts/predict.py --input data/raw/bank-additional-full.csv \\
        --output reports/metrics/call_list.csv

Loads a persisted artifact, validates the input against the contract recorded in
that artifact's metadata, and writes a ranked list with probabilities and priority
tiers. Nothing is fitted here — that is the whole point of the artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from term_deposit.cli._shared import fail, resolve_config
from term_deposit.config import ConfigError
from term_deposit.inference.predictor import PredictorError, load_predictor
from term_deposit.utils.logging import get_logger

app = typer.Typer(help=__doc__, no_args_is_help=False, add_completion=False)
logger = get_logger("cli.predict")


@app.command()
def main(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="CSV of customers to score.")],
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write the ranked call list.")
    ] = None,
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="Config files, merged left to right."),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override a config key: dotted.key=value."),
    ] = None,
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Artifact directory name, or 'latest'.")
    ] = None,
    separator: Annotated[str, typer.Option("--sep", help="Input CSV delimiter.")] = ";",
    id_column: Annotated[
        str | None, typer.Option("--id-column", help="Column to carry through as customer_id.")
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Keep only the highest-scoring N customers.")
    ] = None,
    capacity: Annotated[
        float | None,
        typer.Option("--capacity", help="Keep the top fraction of the list, e.g. 0.2 for 20%."),
    ] = None,
) -> None:
    """Score customers and emit a ranked call list."""
    try:
        app_config = resolve_config(config, set_, extra_defaults=(Path("configs/inference.yaml"),))
    except ConfigError as error:
        fail(str(error))
        return

    if not input_path.is_file():
        fail(f"input file not found: {input_path}")
        return

    try:
        predictor = load_predictor(
            app_config.paths.artifacts_dir, app_config.inference, model_id=model_id
        )
    except PredictorError as error:
        fail(str(error))
        return

    frame = pd.read_csv(input_path, sep=separator)
    logger.info("Read %d row(s) from %s", len(frame), input_path)

    limit = top_k
    if limit is None and capacity is not None:
        if not 0.0 < capacity <= 1.0:
            fail("--capacity must be in (0, 1]")
            return
        limit = max(1, round(len(frame) * capacity))

    try:
        ranked = predictor.rank(frame, top_k=limit, id_column=id_column)
    except (ValueError, PredictorError) as error:
        fail(str(error))
        return

    metadata = predictor.artifact.metadata
    typer.echo("")
    typer.echo(f"Model     {metadata.model_name}  (trained {metadata.created_at})")
    typer.echo(f"Protocol  {metadata.split_strategy}   threshold {predictor.threshold:.4f}")
    typer.echo(f"Scored    {len(frame):,} customer(s); returning {len(ranked):,}")
    typer.echo("")
    typer.echo(f"  mean probability   {ranked['subscription_probability'].mean():.4f}")
    typer.echo(f"  flagged for call   {int(ranked['predicted_class'].sum()):,}")
    if "tier" in ranked.columns:
        counts = ranked["tier"].value_counts().sort_index()
        typer.echo("  tiers              " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output_path, index=False)
        typer.echo("")
        typer.echo(f"Call list written to {output_path}")
    else:
        typer.echo("")
        typer.echo(ranked.head(10).to_string(index=False))

    if metadata.notes:
        typer.echo("")
        for note in metadata.notes:
            typer.echo(f"note: {note}")


if __name__ == "__main__":
    app()
