# ADR 0007: Hand-rolled table schema, Pydantic at the record edge

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

This project has two data boundaries with different shapes.

The first is training-time ingestion: one fixed table of 41,188 rows and 21 columns, read once from
disk. What matters there is checking the whole table cheaply and reporting *every* violation in one
pass, so a contributor fixing a broken file sees the full list instead of one error at a time. The
rules involved are narrow — required columns, row count, nullability, category membership, broad
numeric bounds.

The second is inference: individual customer records arriving from a caller. What matters there is
typed, serialisable request and response contracts, per-field errors a caller can act on, and a hard
refusal of fields that must never be scored.

Forcing one tool across both boundaries means either materialising 41,188 Python objects to validate a
table, or hand-rolling serialisation, aliasing and error formatting for records. Pydantic is already a
dependency because configuration uses it, so the record side costs nothing extra.

## Decision

**Training table — dependency-free.** `src/term_deposit/data/schema.py` defines `ColumnSpec` and
`TableSchema` as plain dataclasses, with `RAW_SCHEMA` built from the vocabulary in
`src/term_deposit/constants.py`. Validation is vectorised over pandas Series, accumulates all
violations, and raises a single `SchemaValidationError` listing them.

**Inference records — Pydantic.** `src/term_deposit/inference/schema.py` defines `CustomerRecord`,
`ScoringRequest`, `ScoredCustomer` and `ScoringResponse` as Pydantic models. `CustomerRecord` is
`frozen=True` with `extra="forbid"`, so `duration` or a target column sent by a caller is rejected at
the boundary rather than silently ignored. Field aliases preserve the raw dotted names
(`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`) so `to_row()` produces exactly the
column names the fitted pipeline expects.

## Consequences

Positive:

- Training validation stays small, vectorised and free of an extra dependency.
- All table violations surface together instead of one per run.
- Inference gets typed errors, JSON schema and alias handling for free.
- `extra="forbid"` makes the post-call leakage mistake impossible to make quietly at the API edge.
- Each boundary is exercised by its own tests without dragging in the other's machinery.

Negative:

- Category domains and numeric bounds are expressed twice, in two forms.
- A schema change requires touching both `constants.py`/`TableSchema` and the Pydantic record; nothing
  enforces that the two stay in agreement beyond tests.
- The hand-rolled schema has no statistical checks, no drift assertions and no report rendering.
- `validate_frame()` on the inference path checks a DataFrame in Python and would need profiling before
  being used on very large batches.
- Two validation vocabularies is more to explain to a new reader than one.

## Alternatives considered

- **Pydantic for the training table too:** rejected. Materialising 41,188 model instances to check
  column membership is slower and less clear than a vectorised pass, and it produces per-row errors
  where a per-column summary is what a data fix needs.
- **Pandera or Great Expectations for both:** rejected for now. They are the right answer once the rule
  set grows to statistical checks, conditional expectations or a validation report; today they would
  add a dependency to express five rule types.
- **Hand-rolled validation for inference as well:** rejected. It would mean reimplementing aliasing,
  coercion, error formatting and JSON schema that Pydantic already provides and configuration already
  depends on.
- **No explicit contract, rely on scikit-learn errors:** rejected. Failures would appear deep inside a
  transformer with an unhelpful message, long after the point where the bad input entered.
- **One schema definition generating both:** rejected as machinery disproportionate to two small
  contracts that are deliberately different in shape.
