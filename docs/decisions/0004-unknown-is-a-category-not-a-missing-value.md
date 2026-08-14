# ADR 0004: `unknown` is a category, not a missing value

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

Several UCI categorical fields contain the literal value `unknown`. Pandas does not parse it as null,
and the source distinguishes it from an absent cell. It may indicate refusal, unavailable documents,
legacy CRM capture, or another unrecorded process.

Blind mode imputation would convert that process into a guessed category. Dropping rows would change
the population and could disproportionately remove customers for whom administrative data is sparse.
Treating it as a normal level allows the pipeline to score the source as published.

## Decision

Keep `unknown` in the declared category domains for job, marital status, education, default, housing
and loan. One-hot encoding preserves a separate indicator. Data-quality reports count literal
`unknown` values separately from true nulls.

Reject actual nulls at the raw-data and inference boundaries. Set `OneHotEncoder(handle_unknown="ignore")`
for genuinely new category labels during matrix transformation, while Pydantic inference validation
continues to reject values outside the declared contract.

## Consequences

Positive:

- No undocumented imputation assumption enters the model.
- The full published population remains available.
- Quality reports distinguish collection gaps from missing cells.
- The artifact can reproduce training transformations exactly.

Negative:

- `unknown` may act as a proxy for protected or disadvantaged groups.
- Its meaning can change when the upstream CRM process changes.
- A score may depend on data quality rather than customer propensity.
- The model needs a subgroup audit before operational use.

## Alternatives considered

- **Mode imputation:** rejected because it invents a category and hides the collection process.
- **Drop rows or columns:** rejected because it changes coverage without evidence that the omission is
  random.
- **Map to null and use a generic imputer:** rejected for the same loss of source semantics.
- **Keep as a category:** accepted as the least assumptive representation, with an explicit fairness
  caveat.
