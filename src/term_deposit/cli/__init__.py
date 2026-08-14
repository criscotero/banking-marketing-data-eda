"""Command-line interface.

All four commands are implemented in this package rather than in ``scripts/`` so
that they are importable, unit-testable with typer's ``CliRunner``, and available
inside a container as a single ``term-deposit`` executable. The files under
``scripts/`` are thin shims over these commands and remain the documented
interface.

Each command module also exposes its own single-command ``app``, which is what
the shim invokes. Here the underlying functions are registered directly on the
root app: mounting a single-command sub-app with ``add_typer`` would require
callers to type a redundant ``term-deposit train main``.
"""

from __future__ import annotations

import typer

from term_deposit import __version__
from term_deposit.cli.evaluate import main as evaluate_command
from term_deposit.cli.predict import main as predict_command
from term_deposit.cli.prepare_data import main as prepare_data_command
from term_deposit.cli.train import main as train_command

app = typer.Typer(
    name="term-deposit",
    help="Pre-call propensity scoring for bank term-deposit campaigns.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("prepare-data", help="Download, verify and profile the dataset.")(prepare_data_command)
app.command("train", help="Train, compare and persist models.")(train_command)
app.command("evaluate", help="Re-evaluate a persisted artifact.")(evaluate_command)
app.command("predict", help="Score customers into a ranked call list.")(predict_command)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


__all__ = ["app"]


if __name__ == "__main__":  # pragma: no cover
    app()
