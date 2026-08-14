# ADR 0002: Out-of-time evaluation is the primary protocol

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

The common evaluation for Bank Marketing is a stratified random split. Here that protocol mixes rows
from the same contact months across train and test. Because macro variables identify the month and the
response rate changes by month, it tests interpolation across known periods rather than scoring the
next campaign period.

The source has `month` but no year. Rows remain in contact order, so a decrease in month number marks a
year rollover and reconstructs May 2008 through November 2010.

Under the random protocol, Random Forest reports pooled ROC-AUC 0.8116 but only 0.5865 within period.
Under the fixed out-of-time window it reports AP 0.6807 against a 0.5214 base rate and ROC-AUC 0.6993.

## Decision

Make `out_of_time` the default split. Train through 2009-05, reserve 2009-06 through 2009-12 for
calibration and threshold choice, and test on 2010-03 through 2010-11.

Retain the random split as a comparison protocol because its contrast demonstrates the leakage risk.
Run a nine-period expanding-window backtest and prefer mean backtest AP for model selection. The
reconstructed period remains a split/report key and never becomes a feature.

## Consequences

Positive:

- Evaluation matches the direction of time at a future campaign.
- Calibration and threshold selection remain separate from test.
- Rolling folds reveal month-to-month variability instead of hiding it in one cut-off.
- The random result remains reproducible without being presented as deployment performance.

Negative:

- Train, validation and test base rates differ sharply: 0.0671, 0.3909 and 0.5214.
- The final test contains only 2,058 rows, making estimates noisier.
- Calendar reconstruction depends on the original contact order.
- Historical performance still does not establish current performance.

## Alternatives considered

- **Random split only:** rejected because it leaks period identity across the boundary.
- **Drop the calendar analysis:** rejected because it would preserve an attractive but misleading
  headline metric.
- **One final holdout only:** retained as a report, but insufficient for selection because it is one
  unusual regime.
- **Use the reconstructed date as a feature:** rejected because absolute future period identity is not
  a stable customer attribute.
