# Data contract

Nothing under this directory is committed except `.gitkeep` files and this README. The source is public,
small and reproducible, so the repository keeps the download logic, the checksum and the schema rather
than a second copy of the data. That keeps a data change distinguishable from a code change and keeps
Git history free of generated products. See
[ADR 0001](../docs/decisions/0001-dataset-is-not-committed.md).

## Directory layout

| Directory | Contract |
|---|---|
| `raw/` | The unmodified extracted source, `bank-additional-full.csv`. Never edited by hand. |
| `interim/` | Regenerated inspection outputs: `dataset_profile.json`, `data_quality.csv`, `period_summary.csv`, `macro_period_collinearity.csv`, `sample.csv`. |
| `processed/` | Reserved for derived datasets that cannot be rebuilt cheaply in memory. Currently empty; the pipeline builds its features inside the fitted estimator. |

`.gitignore` excludes `data/raw/*`, `data/interim/*` and `data/processed/*`.

## Obtaining the data

```bash
make data                                                    # download, verify, profile
uv run python scripts/prepare_data.py --from-file /path/to/bank-additional-full.csv   # offline
```

`scripts/prepare_data.py` downloads the UCI archive, extracts only `bank-additional-full.csv`, verifies
its SHA-256, validates it against the schema and writes the `interim/` profiles. The expected checksum
of the extracted CSV is:

```text
74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
```

It is pinned as `RAW_SHA256` in `src/term_deposit/constants.py`. A mismatch fails the run rather than
silently shifting every reported metric.

## Schema contract

`src/term_deposit/data/schema.py` declares `RAW_SCHEMA` as a `TableSchema` of 21 `ColumnSpec` entries.
Column names, category domains and broad numeric bounds come from `src/term_deposit/constants.py`.
Validation expects 41,188 rows, 21 source columns, a semicolon delimiter, no nulls, and only declared
category values, and it collects every violation before raising `SchemaValidationError`.

## Raw columns

| Column | Type | Meaning |
|---|---|---|
| `age` | integer | Customer age in years. |
| `job` | category | Occupation group. |
| `marital` | category | Marital status; the source description notes that `divorced` covers divorced or widowed. |
| `education` | category | Highest recorded education group. |
| `default` | category | Whether credit is in default. |
| `housing` | category | Whether the customer has a housing loan. |
| `loan` | category | Whether the customer has a personal loan. |
| `contact` | category | Contact communication type: `cellular` or `telephone`. |
| `month` | category | Month of the last contact in this campaign, as a three-letter abbreviation. |
| `day_of_week` | category | Weekday of the last contact, `mon` through `fri`. |
| `duration` | integer | Last-call duration in seconds. Known only after the call ends. |
| `campaign` | integer | Contacts during the current campaign, including the current contact. |
| `pdays` | integer | Days since the prior campaign contact; `999` is the sentinel for never contacted. |
| `previous` | integer | Contacts before this campaign. |
| `poutcome` | category | Outcome of the previous marketing campaign. |
| `emp.var.rate` | float | Quarterly employment variation rate. |
| `cons.price.idx` | float | Monthly consumer price index. |
| `cons.conf.idx` | float | Monthly consumer confidence index. |
| `euribor3m` | float | Daily three-month Euribor rate. |
| `nr.employed` | float | Quarterly number of employees, in thousands. |
| `y` | category | Target: whether the customer subscribed to a term deposit. |

Two fields are derived, not source columns: the loader builds the binary label `subscribed` from `y`,
and the calendar reconstruction builds `contact_period` from row order and the month sequence.

## Excluded from modelling

- `y` and the derived `subscribed` are labels, not features.
- `duration` is dropped in `prepare_dataset` before splitting. It is only known once the call has
  happened, and the model exists to decide who to call, so using it would leak the outcome.
- `contact_period` is a split and reporting key only. It is never passed to an estimator, because an
  absolute future period identity is not a portable customer attribute.
- That leaves 19 features in the default `all` feature set: 9 numeric and 10 categorical.
- The `client_only` feature set additionally excludes the five macro variables, so their contribution
  can be measured as an ablation. See
  [ADR 0003](../docs/decisions/0003-macro-features-are-a-calendar-proxy.md).

Literal `unknown` values are kept as a real category and counted separately from nulls in
`interim/data_quality.csv` ([ADR 0004](../docs/decisions/0004-unknown-is-a-category-not-a-missing-value.md)).
`pdays == 999` is expanded inside the pipeline into a never-contacted flag plus an elapsed-days column,
so the sentinel is not scaled as if 999 days had passed.

## Temporal profile

The reconstruction yields 26 observed contact periods, 2008-05 through 2010-11. The subscription rate
moves from 3.1% in 2008-05 to 57.5% in 2010-05 — the drift that makes a random split misleading. Four
macro fields (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`) have 100% of their
variance between periods, and `euribor3m` has 99.96%, which makes the macro block a near-exact period
identifier. Both tables are regenerated by `make data` as `interim/period_summary.csv` and
`interim/macro_period_collinearity.csv`.

This is why the default evaluation protocol is out-of-time and why pooled metrics are always reported
beside within-period metrics.

## Attribution and licence

Moro, S., Cortez, P. and Rita, P. (2014), "A Data-Driven Approach to Predict the Success of Bank
Telemarketing", *Decision Support Systems*. Published in the UCI Machine Learning Repository at
<https://archive.ics.uci.edu/dataset/222/bank+marketing> under CC BY 4.0. This repository's own MIT
licence does not replace the dataset's attribution terms.
