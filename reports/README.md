# Generated reports

Everything here is a reproducible output of `make train` and `make evaluate`, and everything except
this README and the `.gitkeep` files is gitignored. Two directories:

| Directory | Contents |
|---|---|
| `metrics/` | JSON model reports, CSV comparisons, per-period breakdowns, backtests and tracking records. |
| `figures/` | PNG plots written by `make evaluate`: `roc_curves`, `precision_recall_curves`, `lift_curve`, `calibration`, `confusion_matrix`, `threshold_sweep`, `within_vs_pooled`. |

## Naming

`<protocol>` is `out_of_time` or `random`. `<run_id>` is `<UTC timestamp>__<protocol>__<feature_set>`.

| Pattern | Meaning |
|---|---|
| `<protocol>__<model>.json` | Full metrics for one candidate under one protocol. |
| `<protocol>__comparison.csv` | Flattened one-row-per-candidate comparison. The table to read first. |
| `<protocol>__comparison.json` | The same comparison in JSON. |
| `<protocol>__within_period.csv` | Per-model, per-period ranking metrics. |
| `<protocol>__backtest.csv` | Per-model rolling-origin folds, one row per fold period. |
| `<run_id>.json` | Run-level record: configuration, split sizes and boundaries, drift table, summary. |
| `runs.jsonl` | Append-only index, one line per run. |
| `runs/<run_id>.json` | The detailed tracker record for a single run. |

Re-evaluation writes `evaluation__<model>__<split>.json`. `make predict` writes a ranked call list to
`metrics/call_list.csv`.

## Reading a comparison table

**Always compare `average_precision` against `base_rate`.** No-skill AP equals the positive rate, so
the number means nothing on its own. AP 0.68 against a base rate of 0.52 is a *smaller* improvement
than AP 0.47 against a base rate of 0.11, even though the first number looks better.

**Always read `within_period_roc_auc` next to `roc_auc`.** Their difference is `roc_auc_inflation`. A
large positive value means the pooled score is being earned by ordering calendar periods rather than by
ordering customers within a campaign — which is not what a campaign can act on. Under the recorded
random protocol this is +0.2251 for Random Forest: pooled ROC-AUC 0.8116 against within-period 0.5865.

`precision_at_0.20` is the positive share of the top-scoring fifth of the list; `lift_at_0.20` divides
that by base rate. These are the capacity-limited view that matches how a call centre actually consumes
a ranking. `ece` and `brier_score` describe probability calibration, not ranking quality.

`threshold`, `precision`, `recall` and `f1` all depend on a threshold selected on validation, under the
configured objective. They should not be used to pick a model independently of that objective. Under
the recorded out-of-time run the expected-value threshold calls every test row, which makes those four
columns uninformative there; the capacity metrics remain the useful decision view.

`backtest_ap_mean` is the model-selection metric when every candidate has rolling folds. Read the
individual rows in `<protocol>__backtest.csv` before treating a small difference in the mean as stable.

## Regeneration

```bash
make train               # out-of-time reports and artifact (the default protocol)
make train-random        # random-protocol comparison
make train-client-only   # macro-block ablation
make compare-protocols   # both training protocols, back to back
make evaluate            # re-check the latest artifact and render figures
```

`make evaluate` needs both the raw dataset and a saved artifact. `make clean` removes generated reports
and artifacts while leaving the raw source in place; the checked-in `.gitkeep` files keep the empty
directories.

Do not hand-edit a value in this directory into a preferred result. Change configuration, rerun, and
keep the new run's metadata. Nothing here is production monitoring or evidence of a deployed service.
