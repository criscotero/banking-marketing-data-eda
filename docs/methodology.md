# Methodology

## Problem framing

The operational question is not “will this customer subscribe at a default threshold?” It is “given
the number of calls the centre can place, which customers should appear first?” That makes this a
ranking problem under a capacity constraint.

The main evaluation measure is average precision (AP), supported by precision and lift at fixed list
fractions. Hard classifications are reported only at a threshold chosen on validation. The workflow
does not estimate the causal effect of calling and must not be read as a treatment policy.

## Data and contract

The source is the UCI Bank Marketing `bank-additional-full.csv`: 41,188 contacts, 20 predictors and
the `y` target. Its extracted SHA-256 is:

```text
74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
```

`term_deposit.data.schema.RAW_SCHEMA` checks the row count, required columns, numeric bounds,
categorical domains, and nullability at ingestion. The checksum protects the reported results from an
upstream file changing silently. See [the data contract](../data/README.md) for the column-level table.

Literal `unknown` values are source categories, not pandas nulls. They may encode non-response or CRM
collection practices, so the pipeline retains them instead of imputing a guessed value. That choice
also creates a fairness obligation because missing administrative data may not be distributed evenly.

The numeric value `pdays == 999` means the customer was never contacted in a previous campaign. It is
not a duration. `PdaysSentinelEncoder` replaces it with `pdays_never_contacted` plus
`pdays_days_since_contact`, using zero for the latter when the flag is set.

## Calendar reconstruction

The raw data gives the contact month but omits the year. Its rows remain in contact order. Calendar
reconstruction maps month names to integers, starts in 2008, and increments the year whenever the
current month number is smaller than the previous one:

```text
month number decreases -> year rollover
otherwise              -> same year
```

This reconstructs 26 observed periods from May 2008 through November 2010. Missing calendar months do
not matter because only observed contact periods are used. The strict guard rejects reconstructions
with more than 120 distinct periods, the signature produced when a large extract has been shuffled.
It catches gross disorder; it cannot prove that every row is perfectly sorted.

`contact_period` is used only for splitting, rolling folds, drift diagnostics and within-period
metrics. It is never included in `FeatureConfig.input_columns()` and never reaches an estimator.

## Leakage analysis

### Post-outcome leakage: `duration`

Call duration is available only after contact. Because the model exists to decide whom to call, the
value cannot exist at scoring time. `prepare_dataset()` drops it before splitting so no downstream
stage can reintroduce it accidentally. The original notebook already recognised this leak.

### Period leakage: the macro block

The refactor adds a second finding: the five macro variables act as a calendar key in this extract.
The variance share below is `1 - within_period_variance / total_variance`.

| Feature | Between-period variance share | Distinct values | Periods |
|---|---:|---:|---:|
| `emp.var.rate` | 1.000000 | 10 | 26 |
| `cons.price.idx` | 1.000000 | 26 | 26 |
| `cons.conf.idx` | 1.000000 | 26 | 26 |
| `euribor3m` | 0.999648 | 316 | 26 |
| `nr.employed` | 1.000000 | 11 | 26 |

The outcome rate also changes sharply by period:

| Period | Contacts | Subscription rate |
|---|---:|---:|
| 2008-05 | 7,763 | 0.0309 |
| 2008-06 | 4,374 | 0.0430 |
| 2008-07 | 6,685 | 0.0609 |
| 2008-08 | 5,175 | 0.0524 |
| 2008-10 | 67 | 0.6269 |
| 2008-11 | 3,616 | 0.0525 |
| 2008-12 | 10 | 0.1000 |
| 2009-03 | 282 | 0.4468 |
| 2009-04 | 2,458 | 0.1798 |
| 2009-05 | 5,794 | 0.0904 |
| 2009-06 | 715 | 0.3692 |
| 2009-07 | 178 | 0.3708 |
| 2009-08 | 770 | 0.3429 |
| 2009-09 | 267 | 0.3970 |
| 2009-10 | 447 | 0.4027 |
| 2009-11 | 357 | 0.4706 |
| 2009-12 | 172 | 0.5116 |
| 2010-03 | 264 | 0.5682 |
| 2010-04 | 174 | 0.5575 |
| 2010-05 | 212 | 0.5755 |
| 2010-06 | 229 | 0.4672 |
| 2010-07 | 311 | 0.5659 |
| 2010-08 | 233 | 0.5150 |
| 2010-09 | 303 | 0.4950 |
| 2010-10 | 204 | 0.4559 |
| 2010-11 | 128 | 0.4531 |

The 62.69% rate in October 2008 comes from only 67 records; the high-volume change from 3.09% in May
2008 to 57.55% in May 2010 is the more useful illustration.

Under a random split, rows from the same month occur in training and test. A model can associate the
macro block with each month's outcome prevalence and reproduce that prevalence on held-out rows. A
live monthly batch has one current macro context, so those variables cannot perform the same
between-month separation inside that list.

The diagnostic is the gap between a pooled ROC-AUC and the row-weighted average of ROC-AUC computed
within each eligible month:

| Model | Random pooled ROC-AUC | Random within-period ROC-AUC | Inflation |
|---|---:|---:|---:|
| Random Forest | 0.8116 | 0.5865 | +0.2251 |
| XGBoost | 0.8105 | 0.5919 | +0.2186 |
| Gradient Boosting | 0.8104 | 0.5943 | +0.2161 |
| Logistic Regression | 0.8002 | 0.5575 | +0.2427 |
| Prior baseline | 0.5000 | 0.5000 | 0.0000 |

The project reports the gap instead of silently deleting macro variables. The `client_only` feature
set permits a direct ablation when needed; it excludes the macro block but retains client and campaign
history variables.

## Evaluation protocols

### Stratified random split

The random protocol uses 65% training, 15% validation and 20% test, preserving the overall 11.26%
positive rate. It reproduces the common evaluation setup and measures in-distribution pooled ranking.
It does not measure future-period generalisation because all observed calendar regimes are mixed.

| Model | Base rate | AP | ROC-AUC | Within-period ROC-AUC | Lift@20% |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.1126 | 0.4665 | 0.8116 | 0.5865 | 3.2858 |
| XGBoost | 0.1126 | 0.4663 | 0.8105 | 0.5919 | 3.3128 |
| Gradient Boosting | 0.1126 | 0.4628 | 0.8104 | 0.5943 | 3.2805 |
| Logistic Regression | 0.1126 | 0.4401 | 0.8002 | 0.5575 | 3.2374 |
| Prior baseline | 0.1126 | 0.1126 | 0.5000 | 0.5000 | 0.9696 |

### Out-of-time split

Training ends in May 2009, validation covers June-December 2009, and test covers March-November 2010.
It measures transfer to a later regime, but the one final window remains sensitive to its particular
base rate and contains only 2,058 rows.

| Partition | Rows | Positive rate |
|---|---:|---:|
| Train | 36,224 | 0.0671 |
| Validation | 2,906 | 0.3909 |
| Test | 2,058 | 0.5214 |

| Model | Base rate | AP | ROC-AUC | Within-period ROC-AUC | Lift@20% |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5214 | 0.6807 | 0.6993 | 0.7327 | 1.4571 |
| Logistic Regression | 0.5214 | 0.6264 | 0.6463 | 0.7024 | 1.1545 |
| Gradient Boosting | 0.5214 | 0.6055 | 0.6143 | 0.6482 | 1.2988 |
| XGBoost | 0.5214 | 0.5783 | 0.5780 | 0.6114 | 1.1824 |
| Prior baseline | 0.5214 | 0.5214 | 0.5000 | 0.5000 | 0.9776 |

### Rolling-origin backtest

For each of the last nine eligible months, the estimator is rebuilt on every earlier row and scores
that month. Calibration is not applied in these folds. This mirrors repeated monthly fitting better
than either fixed split and reduces dependence on one test boundary.

| Model | Mean base rate | Mean AP | Mean ROC-AUC | Mean lift@20% |
|---|---:|---:|---:|---:|
| Random Forest | 0.5170 | 0.7417 | 0.7421 | 1.5709 |
| XGBoost | 0.5170 | 0.7297 | 0.7271 | 1.5271 |
| Logistic Regression | 0.5170 | 0.7279 | 0.7281 | 1.5188 |
| Gradient Boosting | 0.5170 | 0.7237 | 0.7249 | 1.5468 |
| Prior baseline | 0.5170 | 0.5170 | 0.5000 | 0.9992 |

Selection prefers this mean AP when all candidates have a backtest. Random Forest is therefore the
persisted model for the recorded out-of-time run.

## Metrics

### Average precision

AP summarises precision across recall levels and focuses on the minority outcome. Its no-skill value
equals the split's positive rate, so `AP=0.68` is meaningful only beside `base_rate=0.52`. This is why
every results table shows both.

### Precision, recall and lift at k

For a fraction `k`, deterministic descending score order selects `round(n * k)` rows, with at least one
row. Precision@k is the positive share in that slice, recall@k is the share of all positives captured,
and lift@k is precision@k divided by the base rate. Position breaks score ties reproducibly.

### ROC-AUC and within-period metrics

ROC-AUC is threshold-free and remains a useful ranking diagnostic. It is not the lead metric because
it gives no visible base-rate reference and, when pooled across periods, rewards ordering high-rate
months above low-rate months. The within-period calculation removes that between-period comparison.

### Calibration

Brier score measures squared probability error. Expected calibration error (ECE) compares observed and
predicted rates in equal-frequency bins. On the random test, candidate ECE is 0.0044-0.0136; on the
out-of-time test it is 0.1319-0.1419. The held-out isotonic map fits the original regime but does not
repair a future prevalence shift.

## Threshold selection

Threshold choice occurs on validation only and is then frozen. Three objectives are implemented:

1. `expected_value`: maximise subscriptions times value minus calls times cost, divided by all scored
   customers.
2. `f1`: give precision and recall equal weight.
3. `top_k`: select exactly the configured capacity fraction.

The recorded run uses `expected_value`, value 100 and cost 5 in arbitrary consistent units. These are
placeholders, not bank accounting. Random Forest's out-of-time artifact records threshold 0.2437. On
the 2010 test all 2,058 scores are at or above it, so the hard prediction calls everyone. That outcome
is honest evidence that a measured capacity rule is required before operational use.

## Class imbalance and calibration

Logistic Regression and Random Forest use balanced class weights; XGBoost derives
`scale_pos_weight` from the training fold only; Gradient Boosting and the prior baseline are unweighted.
The fold-local calculation prevents test prevalence from becoming a training hyperparameter.

Weighting can improve minority ranking while shifting scores away from empirical probabilities. Each
main fitted pipeline is therefore wrapped by a frozen isotonic calibrator learned on validation raw
scores. Preprocessing and the base model are not refitted during calibration.

## Feature importance

The supported model-agnostic method is permutation importance on held-out raw columns using AP. It
keeps a categorical feature together and measures loss in the metric the project optimises.

Native tree importance is retained only as a comparison because it favours continuous or
high-cardinality features. In the historical notebook, Random Forest impurity assigns about 0.133 to
`age`, while Gradient Boosting and XGBoost native gain assign about 0.0216 and 0.0069. This disagreement
is why the current methodology does not present impurity importance as a stable explanation. SHAP is
optional and disabled in the recorded runs.

## Reproducibility

- The extracted dataset is checked against a fixed SHA-256.
- Configuration is stored in frozen Pydantic models and written into artifact metadata.
- Seed 42 is passed to Python, NumPy, splitters and supported estimators.
- Each model and fold receives a fresh pipeline; no mutable preprocessor is shared.
- Top-k tie-breaking is stable by original row position.
- Run IDs contain UTC time, protocol and feature set, with numeric collision suffixes.
- `uv.lock` records dependency resolution; metadata records the actual Python and library versions.
- CI may pin `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to one for repeatable
  reductions.

## Limitations that can change the conclusion

The source covers one Portuguese bank during 2008-2010. Current customer behaviour, contact policy,
interest rates and regulation may differ. No external, contemporary, causal, fairness, or subgroup
validation has been run. The out-of-time window is small and has an unusually high positive rate.

The reconstructed calendar relies on row order. The macro block's proxy behaviour is specific to this
extract. Calibration fails across the observed regime shift. Hyperparameters were configured manually,
and the expected-value inputs are placeholders. These limits prevent a deployment claim and require a
fresh evaluation before any operational trial.
