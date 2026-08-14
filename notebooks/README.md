# Notebooks

Notebooks in this repository exist for exploration and communication. They are not the training,
evaluation or inference path. Everything that has to be reproducible runs through the package under
`src/term_deposit/` and is driven by `scripts/` and the Makefile.

The rule that keeps the two apart: **reusable logic lives in `src/term_deposit/`**. A notebook may
explore, plot and narrate, but as soon as a cell contains a function another notebook or a script would
want, that function moves into the package with a test beside it. Notebooks then import the production
module rather than keeping a parallel copy of preprocessing, splitting or metric code. This is what
prevents the notebook and the pipeline from drifting into two different definitions of the same number.

## Inventory

| Notebook | Purpose |
|---|---|
| `exploratory/00-original-analysis.ipynb` | The pre-refactor notebook, kept as a historical record of the starting point. Its conclusions are superseded by the current evaluation; read it as provenance, not as a result. |
| `exploratory/01-data-and-leakage-analysis.ipynb` | Exploratory data analysis together with the calendar reconstruction and the leakage analysis: how `contact_period` is recovered from contact order, and how the macro block behaves as a period proxy. |
| `exploratory/02-model-evaluation.ipynb` | Protocol comparison and error analysis: random against out-of-time, pooled against within-period metrics, threshold behaviour and where the ranking fails. |

Notebook `00` reflects the original assumptions and dependency set. Its code and narrative are
unmodified so the refactor has a visible before-state; only its stored outputs were removed, by the
same `nbstripout` rule that applies to every notebook here (that alone took the file from 2.3 MB to
about 60 KB). It is not maintained against the current package API and is excluded from
`make notebooks-check`.

## Working convention

1. Load settings with `term_deposit.config.load_config` rather than hard-coding paths or seeds.
2. Get validated data and calendar labels from `term_deposit.pipelines.experiment.prepare_dataset`.
3. Read generated tables under `reports/metrics/` instead of transcribing numbers into markdown cells.
4. Move any function worth reusing into `src/term_deposit/` and add a small test for it.
5. Keep every stated conclusion consistent with the out-of-time and within-period evidence.

## Outputs are not committed

The `nbstripout` pre-commit hook strips cell outputs on commit. A clone therefore contains code cells
with no stored figures, tables or execution state, which keeps diffs readable and keeps the repository
free of megabytes of embedded PNGs. The consequence is that **notebooks must be run locally to see
their output** — nothing here renders on its own from a fresh checkout.

## Running them

```bash
uv sync --extra notebooks --extra viz
make notebooks
```

`make notebooks` launches JupyterLab against this directory. Both analysis notebooks need the raw
dataset, so run `make data` first. Neither reads `reports/`: notebook `01` works from
`prepare_dataset`, and notebook `02` calls `run_experiment` itself with `persist=False`, so it
trains its own models and does not overwrite the artifacts `make train` produced. Notebook `02`
trains fifteen models across three protocols and takes several minutes.

To execute the analysis notebooks end to end and fail on any error — the check that catches a notebook
which no longer matches the package API:

```bash
make notebooks-check
```

`notebooks-check` covers `01` and `02` only. Notebook `00` is excluded because it is a historical
artefact. The supported executable path for results remains `make data`, `make train`, `make evaluate`
and `make predict`.

## Review checklist

Before committing a notebook, confirm:

- Call `duration` and the target never appear as input features.
- The calendar label `contact_period` is used for splitting and reporting only, never passed to an
  estimator.
- Average precision is always shown next to the base rate.
- Pooled ROC-AUC is always shown next to within-period ROC-AUC.
- Thresholds shown as decisions were selected on validation, not on the test set being displayed.
- Every number comes from a generated report or is computed in a visible cell.
- Any deployment or architecture sketch is labelled as a proposal; nothing here is deployed.
