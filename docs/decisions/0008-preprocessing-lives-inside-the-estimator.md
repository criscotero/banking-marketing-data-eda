# ADR 0008: Preprocessing lives inside the estimator

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

The original notebook constructed one `ColumnTransformer` and passed the same mutable object to four
scikit-learn `Pipeline` instances. Pipelines keep that reference rather than copying it. A later
`fit_transform` could therefore refit preprocessing shared by already-trained models and make results
depend on cell execution order.

Preprocessing outside an estimator also creates a second inference implementation and makes it easy to
fit scaling or category state on held-out rows.

## Decision

`build_preprocessor()` returns a fresh pipeline every time. `build_pipeline()` appends one estimator
to that fresh selection, sentinel encoding, scaling and one-hot path. Cross-validation and backtest
clone or rebuild the complete estimator for each fold.

Persist the fitted whole, not an estimator alone. When calibration is enabled, persist the frozen
calibrated wrapper around that fitted pipeline because it is the scorer that produced reported
probabilities.

## Consequences

Positive:

- Preprocessing learns only from each training fold.
- Models cannot mutate one another's fitted transformations.
- Training, evaluation and inference execute the same code path.
- The artifact carries column order, unseen-category behaviour and `pdays` semantics.

Negative:

- Each candidate owns a separate fitted preprocessor and uses more memory.
- Joblib artifacts are tied to compatible Python and library versions.
- Inspecting the final estimator requires navigating the pipeline or calibrator wrapper.
- A transform change requires retraining; it cannot be patched beside an old model.

## Alternatives considered

- **One shared fitted transformer:** rejected because mutable shared state caused the original bug.
- **Pre-transform the full table:** rejected because it leaks held-out statistics into folds.
- **Duplicate transforms in inference:** rejected because the two paths will drift.
- **Persist only the estimator:** rejected because it omits the input-to-matrix contract.
