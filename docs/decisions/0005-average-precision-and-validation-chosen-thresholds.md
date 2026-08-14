# ADR 0005: Use average precision and validation-chosen thresholds

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

The full dataset has an 11.27% positive class. A telephone campaign consumes the highest-ranked part
of a list, making minority ranking and top-k yield more direct objectives than overall accuracy.
ROC-AUC remains useful but has no visible no-skill base rate and is inflated by period separation here.

The original notebook compared models at threshold 0.5. Logistic Regression and Random Forest used
class weights, XGBoost used `scale_pos_weight`, and Gradient Boosting was unweighted. A fixed 0.5
therefore compared different score scales rather than equivalent operating choices.

## Decision

Use average precision as the primary metric, always reported beside base rate. Report precision and
lift at configured capacity fractions and retain pooled plus within-period ROC-AUC as diagnostics.

Choose one threshold per model on validation only, then freeze it for test. Support `expected_value`,
`f1`, and `top_k` objectives. The default expected-value rule exposes placeholder economics of 100
units per subscription and 5 per call instead of hiding them in threshold 0.5.

## Consequences

Positive:

- AP has an explicit no-skill reference equal to prevalence.
- Lift and precision map directly to a capacity-limited campaign.
- Test labels do not tune the operating point.
- Weighted and unweighted estimators receive objective-specific thresholds.

Negative:

- AP changes with prevalence and cannot be compared across populations without the base rate.
- Placeholder economics can produce an unrealistic decision; in the 2010 test the chosen Random
  Forest threshold 0.2437 marks every row.
- A threshold is not portable across calibration or regime changes.
- F1 and expected value encode different business preferences.

## Alternatives considered

- **Accuracy or fixed 0.5:** rejected because both obscure imbalance and score-scale differences.
- **ROC-AUC only:** rejected as the lead because the period proxy inflates it.
- **Tune on test:** rejected because it would contaminate the reported result.
- **Top-k only:** available and likely operationally preferable once real agent capacity is known.
