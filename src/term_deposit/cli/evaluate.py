"""Re-evaluate a persisted model and render the report figures.

    uv run python scripts/evaluate.py --figures

Loads a saved artifact and scores it against the test split of the protocol it
was trained under. This is the check that the shipped artifact and the published
metrics are the same thing — a report written during training proves nothing
about the file that was saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from term_deposit.cli._shared import fail, resolve_config
from term_deposit.config import AppConfig, ConfigError
from term_deposit.data.loader import DatasetNotFoundError
from term_deposit.data.splits import make_split
from term_deposit.evaluation.report import EvaluationReport, evaluate_artifact
from term_deposit.inference.predictor import Predictor, PredictorError, load_predictor
from term_deposit.pipelines.experiment import prepare_dataset
from term_deposit.utils.logging import get_logger
from term_deposit.utils.serialization import write_json

app = typer.Typer(help=__doc__, no_args_is_help=False, add_completion=False)
logger = get_logger("cli.evaluate")


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
    model_id: Annotated[
        str, typer.Option("--model-id", help="Artifact directory name, or 'latest'.")
    ] = "latest",
    split_part: Annotated[
        str, typer.Option("--split", help="Which partition to score: test, validation or train.")
    ] = "test",
    figures: Annotated[
        bool, typer.Option("--figures", help="Render report figures (needs the viz extra).")
    ] = False,
) -> None:
    """Evaluate a saved artifact."""
    try:
        app_config = resolve_config(config, set_)
    except ConfigError as error:
        fail(str(error))
        return

    try:
        predictor = load_predictor(
            app_config.paths.artifacts_dir, app_config.inference, model_id=model_id
        )
    except PredictorError as error:
        fail(str(error))
        return

    metadata = predictor.artifact.metadata
    if metadata.split_strategy != app_config.split.strategy:
        typer.echo(
            f"note: the artifact was trained under the '{metadata.split_strategy}' protocol "
            f"but the current config selects '{app_config.split.strategy}'. "
            f"Re-evaluating under '{metadata.split_strategy}' to match the artifact."
        )
        app_config = app_config.model_copy(
            update={
                "split": app_config.split.model_copy(update={"strategy": metadata.split_strategy})
            }
        )

    try:
        prepared = prepare_dataset(app_config, download=False)
    except DatasetNotFoundError as error:
        fail(str(error))
        return

    split = make_split(
        prepared.frame, app_config.split, feature_columns=tuple(metadata.input_columns)
    )
    features = {"test": split.X_test, "validation": split.X_validation, "train": split.X_train}
    labels = {"test": split.y_test, "validation": split.y_validation, "train": split.y_train}
    if split_part not in features:
        fail(f"--split must be one of {sorted(features)}")
        return
    if features[split_part].empty:
        fail(f"the '{split_part}' partition is empty under this protocol")
        return

    report = evaluate_artifact(
        predictor.artifact,
        features[split_part],
        labels[split_part],
        app_config,
        periods=split.periods.loc[features[split_part].index],
    )

    output = app_config.paths.metrics_dir / f"evaluation__{metadata.model_name}__{split_part}.json"
    write_json(output, report.to_dict())

    metrics = report.test_metrics
    typer.echo("")
    typer.echo(f"Model     {metadata.model_name}  ({metadata.estimator})")
    typer.echo(f"Trained   {metadata.created_at}  protocol={metadata.split_strategy}")
    typer.echo(f"Partition {split_part}  n={metrics.n_rows:,}  base rate={metrics.base_rate:.4f}")
    typer.echo("")
    typer.echo(f"  average precision      {metrics.average_precision:.4f}")
    typer.echo(f"  ROC-AUC (pooled)       {metrics.roc_auc:.4f}")
    if report.within_period:
        within = report.within_period.get("weighted_roc_auc", float("nan"))
        typer.echo(f"  ROC-AUC (within month) {within:.4f}")
    typer.echo(f"  Brier score            {metrics.brier_score:.4f}")
    typer.echo(f"  calibration error      {metrics.expected_calibration_error:.4f}")
    typer.echo(f"  threshold              {metrics.threshold:.4f}")
    typer.echo(
        f"  precision / recall     {metrics.precision:.4f} / {metrics.recall:.4f} "
        f"(F1 {metrics.f1:.4f})"
    )
    for fraction, values in metrics.top_k.items():
        typer.echo(
            f"  top {float(fraction):>6.0%}          precision={values['precision']:.4f}  "
            f"lift={values['lift']:.2f}  recall={values['recall']:.4f}"
        )
    typer.echo("")
    typer.echo(f"Report written to {output}")

    if figures:
        _render_figures(app_config, predictor, features[split_part], labels[split_part], report)


def _render_figures(
    app_config: AppConfig,
    predictor: Predictor,
    features: pd.DataFrame,
    labels: pd.Series,
    report: EvaluationReport,
) -> None:
    """Render and save the report figures."""
    try:
        from term_deposit.evaluation import plots
    except ImportError as error:
        typer.echo(f"note: skipping figures ({error})")
        return

    scores = {predictor.artifact.metadata.model_name: predictor.predict_proba(features)}
    figures_dir = app_config.paths.figures_dir
    written = []
    try:
        written.append(
            plots.save_figure(plots.plot_roc_curves(scores, labels), figures_dir / "roc_curves.png")
        )
        written.append(
            plots.save_figure(
                plots.plot_precision_recall_curves(scores, labels),
                figures_dir / "precision_recall_curves.png",
            )
        )
        written.append(
            plots.save_figure(plots.plot_lift_curve(scores, labels), figures_dir / "lift_curve.png")
        )
        written.append(
            plots.save_figure(
                plots.plot_calibration(scores, labels), figures_dir / "calibration.png"
            )
        )
        written.append(
            plots.save_figure(
                plots.plot_confusion_matrix(
                    labels,
                    next(iter(scores.values())),
                    report.threshold.threshold,
                    predictor.artifact.metadata.model_name,
                ),
                figures_dir / "confusion_matrix.png",
            )
        )
        written.append(
            plots.save_figure(
                plots.plot_threshold_sweep(report.sweep, report.threshold.threshold),
                figures_dir / "threshold_sweep.png",
            )
        )
        if report.within_period.get("by_period"):
            written.append(
                plots.save_figure(
                    plots.plot_within_vs_pooled([report]), figures_dir / "within_vs_pooled.png"
                )
            )
    except ImportError as error:
        typer.echo(f"note: skipping figures ({error})")
        return

    typer.echo(f"Wrote {len(written)} figure(s) to {figures_dir}")


if __name__ == "__main__":
    app()
