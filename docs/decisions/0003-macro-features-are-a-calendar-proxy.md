# ADR 0003: Report macro features as a calendar proxy

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

The macro block looks like ordinary numeric context, but its values are nearly fixed within contact
month. Between-period variance shares are 1.000000 for `emp.var.rate`, `cons.price.idx`,
`cons.conf.idx`, and `nr.employed`, and 0.999648 for `euribor3m`.

At the same time, monthly response prevalence changes from 0.0309 in May 2008 to 0.5755 in May 2010.
A random split lets the model learn a month's prevalence from macro inputs and replay it on held-out
rows from that month. A monthly campaign batch has one macro context, so that separation disappears.

Measured pooled-minus-within-period ROC-AUC inflation is +0.2251 for Random Forest, +0.2186 for
XGBoost, +0.2161 for Gradient Boosting and +0.2427 for Logistic Regression.

## Decision

Keep the macro block in the default `all` feature set so its contribution remains measurable, but
always report within-period metrics next to pooled metrics. Treat a large gap as period signal, not
customer signal.

Provide `features.feature_set=client_only` to exclude the macro block for an ablation. Do not claim
that macro variables improve within-campaign ranking unless the client-only comparison establishes it
under temporal evaluation.

## Consequences

Positive:

- The central failure mode is visible in the first comparison table.
- The repository can reproduce common pooled results and explain why they are optimistic.
- The macro contribution can be tested without maintaining a second code path.
- Future data can show whether these variables acquire genuine within-period variation.

Negative:

- Users can still quote the larger pooled value without reading its neighbouring diagnostic.
- The default artifact contains period-proxy variables.
- Within-period estimates are noisy for small months and skip periods with fewer than 50 rows.
- An ablation requires another training run and is not included in the recorded reports.

## Alternatives considered

- **Drop macro variables unconditionally:** simpler, rejected because it prevents measuring their
  behaviour and may discard useful signal in a future dataset with within-period variation.
- **Keep them without a diagnostic:** rejected because pooled metrics then overstate the use case.
- **Aggregate to one row per month:** rejected because the operational output is a customer ranking.
- **Feed contact period directly:** rejected as a stronger and less portable version of the same proxy.
