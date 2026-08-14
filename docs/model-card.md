# Model card: term-deposit propensity

This card describes the model persisted at `artifacts/20260813T194016Z__out_of_time__all/`. Every
number below is read from that artifact's `metadata.json` or from `reports/metrics/`. Nothing here
is deployed.

## Model details

| Item | Value |
|---|---|
| Name / task | `random_forest`; binary propensity scoring, used as a ranking function |
| Estimator | `sklearn.ensemble.RandomForestClassifier` |
| Hyperparameters | `n_estimators=400`, `max_depth=12`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| Preprocessing | `ColumnSelector` -> `PdaysSentinelEncoder` -> `ColumnTransformer` with `StandardScaler` on numeric and `OneHotEncoder(handle_unknown="ignore")` on categorical columns — all inside the same `sklearn.pipeline.Pipeline` as the estimator |
| Calibration | `FrozenPipelineCalibrator`, isotonic, fitted on the validation split only; the fitted pipeline is a fixed scorer and is not refitted |
| Input columns | 19 raw columns: `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome` |
| Output | Positive-class probability, plus a hard class at the recorded threshold and a rank-based priority tier |
| Selection protocol | `out_of_time`; the winner was chosen on mean rolling-origin backtest average precision |
| Trained | 2026-08-13T19:45:44Z, git revision `f0c8ff4` |
| Environment | Python 3.12.11, scikit-learn 1.9.0, numpy 2.5.2, pandas 2.3.3, xgboost 3.4.0, `term_deposit` 1.0.0 |
| Data checksum | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |
| Licence / author | MIT for the code, UCI Bank Marketing for the data; the author of this repository |

`duration` is excluded from the inputs: it is known only after a call ends, and the model exists to
decide whom to call.

### Decision threshold

| Item | Value |
|---|---|
| Threshold | 0.2437 (`0.2436548223350254`) |
| Objective | `expected_value`, chosen on the validation split |
| Assumed value per subscription / cost per call | 100 / 5 |

Those two economic inputs are **placeholders in arbitrary but internally consistent units**, set in
`configs/base.yaml`. They are not measured bank accounting; no monetary claim in this repository
should be read as one. They make the cost of a missed subscriber relative to a wasted call explicit.

The threshold selected 100% of the validation rows, and on the out-of-time test window all 2,058
scores fall above it, so the hard classification calls everyone: precision 0.5214 (equal to the base
rate), recall 1.0, confusion `tn=0, fp=985, fn=0, tp=1073`. That is reported rather than hidden — it
is direct evidence that the expected-value rule under placeholder economics does not produce a usable
operating point across this regime change, and that a capacity rule (`threshold_objective: top_k`
with `capacity_fraction`) would be required before any operational trial.

## Intended use

Ranking a call list **within a single campaign month** so that a limited number of agent hours are
spent on the customers most likely to subscribe. The consumable output is the ordering, optionally
truncated to the number of calls the centre can place: `scripts/predict.py --capacity 0.2`.

Intended users are the author, reviewers of this portfolio, and anyone reproducing the analysis. The
context is offline, batch, human-in-the-loop: a scored list is a suggestion of call order, not a
decision about a customer.

## Out-of-scope use

- **Credit decisions of any kind** — eligibility, pricing, limits, collections. The model predicts a
  marketing response, not creditworthiness.
- **Adverse action.** No score here may be used to deny, withdraw or worsen an offer, and the model
  provides no reason codes suitable for an adverse-action notice.
- **Causal claims.** The data records outcomes under one bank's contact policy. The model cannot
  estimate the effect of calling someone and must not be read as a treatment policy.
- **Any use requiring calibrated absolute probabilities across a regime change.** See the calibration
  finding below: the probabilities are trustworthy in-distribution and are not after a regime shift.
- **Populations unlike Portuguese retail banking customers contacted between 2008 and 2010.** Other
  countries, products, channels or decades.
- **Real-time or per-customer scoring as an authoritative answer.** The artifact is a batch scorer
  with no serving infrastructure, no monitoring and no fairness audit.

## Factors

The model consumes demographic attributes and close proxies — `age`, `job`, `education`, `marital` —
alongside the financial-status fields `default`, `housing`, `loan`, contact history and macro
indicators. Relevant grouping factors for a future evaluation would therefore include age band, job
category, education level, marital status and existing-loan status, plus contact channel and the
calendar period. **None of these were analysed as evaluation subgroups**; the only factor the
reported metrics are broken down by is `contact_period`. The literal value `unknown` is retained as a
category rather than imputed, in `job`, `marital`, `education`, `default`, `housing` and `loan`.

## Metrics

- **Average precision (AP)**, the primary metric. Its no-skill value equals the split's positive
  rate, so every table below prints the base rate next to it.
- **ROC-AUC** as a threshold-free ranking diagnostic, alongside the row-weighted **within-period
  ROC-AUC** per eligible calendar month. The gap is calendar signal, not customer signal.
- **Precision and lift at k** for k in {5%, 10%, 20%, 30%}, ties broken by original row position.
- **Brier score** and **expected calibration error** over 10 equal-frequency bins.
- **Bootstrap intervals** on AP, 500 resamples.

## Evaluation data

The out-of-time protocol splits by reconstructed calendar month; the raw file carries no year column,
so `features/calendar.py` reconstructs it from month rollovers in row order.

| Partition | Calendar window | Rows | Positive rate |
|---|---|---:|---:|
| Train | through 2009-05 | 36,224 | 0.0671 |
| Validation | 2009-06 to 2009-12 | 2,906 | 0.3909 |
| Test | 2010-03 to 2010-11 | 2,058 | 0.5214 |

The test window is small and its base rate is far above the dataset average of 0.1126; the rolling
origin backtest exists because 2,058 rows is one draw from a noisy distribution.

## Training data

UCI Bank Marketing, `bank-additional-full.csv`: 41,188 contacts of a Portuguese retail bank, 20
predictors and a binary `y`. The dataset is not committed; `scripts/prepare_data.py` downloads it and
verifies the SHA-256 above. `data/schema.py::RAW_SCHEMA` checks row count, required columns, numeric
bounds, categorical domains and nullability before anything is fitted. `duration` is dropped before
splitting; `pdays == 999` is a "never contacted" sentinel split into a flag plus an elapsed-days
column; the contact period is used only for splitting, folds and diagnostics.

## Quantitative analyses

### Protocol 1 — random split, pooled test set

65/15/20 stratified train/validation/test. Measures in-distribution pooled ranking. Test n = 8,238.

| Model | Base rate | AP | ROC-AUC | Within-period ROC-AUC | Lift@20% | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.1126 | 0.4665 | 0.8116 | 0.5865 | 3.2858 | 0.0088 |
| XGBoost | 0.1126 | 0.4663 | 0.8105 | 0.5919 | 3.3128 | 0.0044 |
| Gradient Boosting | 0.1126 | 0.4628 | 0.8104 | 0.5943 | 3.2805 | 0.0103 |
| Logistic Regression | 0.1126 | 0.4401 | 0.8002 | 0.5575 | 3.2374 | 0.0136 |
| Prior baseline | 0.1126 | 0.1126 | 0.5000 | 0.5000 | 0.9696 | 0.0000 |

Random Forest AP bootstrap interval: 0.4665, 95% interval 0.4341 to 0.4993, against a no-skill value
of 0.1126. The pooled-to-within-period ROC-AUC gap is +0.2251: under a random split, rows from the
same month appear in both train and test and the macro block acts as a month identifier, so the model
reproduces each month's prevalence on held-out rows — a capability no live monthly batch has.

### Protocol 2 — out-of-time split, single held-out window

Test n = 2,058, covering 2010-03 to 2010-11.

| Model | Base rate | AP | ROC-AUC | Within-period ROC-AUC | Lift@20% | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.5214 | 0.6807 | 0.6993 | 0.7327 | 1.4571 | 0.1419 |
| Logistic Regression | 0.5214 | 0.6264 | 0.6463 | 0.7024 | 1.1545 | 0.1319 |
| Gradient Boosting | 0.5214 | 0.6055 | 0.6143 | 0.6482 | 1.2988 | 0.1377 |
| XGBoost | 0.5214 | 0.5783 | 0.5780 | 0.6114 | 1.1824 | 0.1319 |
| Prior baseline | 0.5214 | 0.5214 | 0.5000 | 0.5000 | 0.9776 | 0.1305 |

Random Forest AP bootstrap interval: 0.6807, 95% interval 0.6530 to 0.7094, against a no-skill value
of 0.5214. Brier 0.2447, log loss 0.6830. Random Forest capacity metrics on this window:

| List fraction | Precision | Recall | Lift |
|---|---:|---:|---:|
| 5% | 0.7670 | 0.0736 | 1.4711 |
| 10% | 0.7621 | 0.1463 | 1.4618 |
| 20% | 0.7597 | 0.2917 | 1.4571 |
| 30% | 0.7455 | 0.4287 | 1.4299 |

The AP of 0.6807 sits only 0.159 above the 0.5214 no-skill line, and the within-period ROC-AUC here
slightly *exceeds* the pooled one (inflation -0.0334) — the expected sign once the between-month
comparison is removed.

### Protocol 3 — rolling-origin backtest, 9 monthly folds

Fit on all prior months, score the next, repeat — one fold per trailing month from 2010-03 to
2010-11. This is the basis on which the persisted model was selected.

| Model | Mean base rate | Mean AP | Mean ROC-AUC | Mean lift@20% |
|---|---:|---:|---:|---:|
| Random Forest | 0.5170 | 0.7417 | 0.7421 | 1.5709 |
| XGBoost | 0.5170 | 0.7297 | 0.7271 | 1.5271 |
| Logistic Regression | 0.5170 | 0.7279 | 0.7281 | 1.5188 |
| Gradient Boosting | 0.5170 | 0.7237 | 0.7249 | 1.5468 |
| Prior baseline | 0.5170 | 0.5170 | 0.5000 | 0.9992 |

Fold-to-fold spread for Random Forest: AP standard deviation 0.0546, ROC-AUC 0.0532, lift@20% 0.1663,
base rate 0.0481 — the same order as the gap between the top four models, which is why the ordering
across those four should not be over-read.

### Calibration

| Protocol | ECE range across the four real models | Random Forest ECE |
|---|---|---:|
| Random split, in-distribution | 0.0044 to 0.0136 | 0.0088 |
| Out-of-time, across a regime change | 0.1319 to 0.1419 | 0.1419 |

Isotonic calibration on held-out validation data works well when the test rows come from the same
mixture of months, and fails by roughly an order of magnitude when the test window's base rate has
moved from 0.1126 to 0.5214. The ranking degrades gracefully across the regime change; the
probabilities do not.

### Cross-validation on the training window

5-fold stratified k-fold on the training split, scored by AP. Random Forest: mean 0.2137, standard
deviation 0.0162 under the out-of-time training window; mean 0.4560, standard deviation 0.0216 under
the random one. The two are not comparable (different base rates) and measure stability, not
temporal generalisation.

## Ethical considerations

The model consumes `age`, `job`, `education`, `marital` and the loan-status fields `default`,
`housing`, `loan`. Several are protected characteristics or close proxies under common equality
frameworks. Ranking on them can systematically deprioritise groups: a customer who is never called
cannot subscribe, so a low rank is a withheld opportunity, and repeated application of the same
ranking compounds that withholding over time.

**No fairness analysis was run.** There is no subgroup performance breakdown, no calibration-by-group
check, no selection-rate comparison at any capacity fraction, and no measure of how the top 5% of a
list distributes across age bands, job categories or education levels. A subgroup audit covering at
least those factors would be required before operational use; this card is not evidence of fairness.

The literal `unknown` category is retained rather than imputed, on the grounds that non-response and
CRM collection gaps are information. That creates an obligation the project has not discharged:
missing administrative data is rarely distributed evenly across a population, so `unknown` may itself
correlate with protected characteristics, and any effect learned from it may be an effect of who
tends to be recorded incompletely. Separately, the 2008-2010 training period covers a financial
crisis in which contact policy, customer behaviour and interest rates were all unusual; behaviour
learned from it should not be assumed to be a stable property of customers.

## Caveats and recommendations

- **Calibration does not survive the regime change.** ECE 0.0044-0.0136 in-distribution versus
  0.1319-0.1419 out of time. Treat out-of-time probabilities as scores for ordering, not as rates,
  and recalibrate on recent labelled data before any expected-value calculation.
- **The out-of-time test window is small.** 2,058 rows, base rate 0.5214 against a dataset average of
  0.1126. Prefer the 9-fold backtest averages when quoting performance.
- **No hyperparameter search was run.** Every value in `configs/training.yaml` was set by hand, and
  the ordering of the top four models is within fold-to-fold noise.
- **No external validation.** One bank, one country, one extract; no contemporary, second-source or
  held-out-institution check.
- **The threshold is not operationally usable as recorded.** It calls 100% of the out-of-time test
  window. Switch `threshold_objective` to `top_k` and set `capacity_fraction` from real agent hours.
- **The economic inputs are placeholders.** 100 and 5, in arbitrary units.
- **The calendar is reconstructed, not given.** Period assignment depends on the extract's row order
  being chronological. A guard rejects reconstructions spanning more than 120 distinct periods, which
  catches gross disorder but cannot prove every row is correctly placed.
- **The macro block behaves as a period key in this extract.** Pooled metrics under a random split
  overstate within-campaign performance by roughly 0.22 ROC-AUC. Quote the within-period number.
- **Nothing is deployed.** There is no endpoint, no monitoring, no shadow run and no A/B test.

### Maintenance

Retraining should be triggered by any of: a new campaign month of labelled outcomes; a shift in the
observed base rate beyond the backtest fold-to-fold spread of 0.0481; a change to the contact policy
or the call-list source; a change in the raw file's SHA-256; or a scikit-learn/xgboost upgrade that
`load_artifact`'s version-drift warning flags. What to monitor, most useful first:

| Signal | Why | Where it is already computed |
|---|---|---|
| Within-period ROC-AUC and AP against the period base rate | The only ranking quality a live monthly campaign experiences | `evaluation.metrics.within_period_metrics`, written to `reports/metrics/*__within_period.csv` |
| Base-rate drift per period | Drives both threshold validity and AP interpretation | `features.calendar.period_summary`; `data.splits.describe_drift` |
| Macro-feature drift | The macro block is a period proxy, so its movement is the regime-change signal | `features.calendar.macro_period_collinearity`; `data.splits.describe_drift` |
| Expected calibration error and Brier on recent labelled batches | Detects the failure mode this card documents | `evaluation.metrics.expected_calibration_error`, `calibration_summary` |
| Realised precision and lift at the operating capacity | Ties the offline number to the campaign outcome | `evaluation.metrics.precision_at_k`, `lift_at_k` |

A subgroup fairness audit is a prerequisite for operational use, not a maintenance activity.
