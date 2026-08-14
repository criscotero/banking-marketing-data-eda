# Architecture decision records

An architecture decision record (ADR) captures a consequential choice, the evidence that was available
when it was made, and the trade-offs accepted alongside it. They exist here so that a reader can tell
which parts of this repository are deliberate and why, rather than reverse-engineering intent from the
code.

ADRs are historical constraints for this repository, not universal recommendations. A decision that
later turns out to be wrong is not edited away; it is superseded by a new ADR that says so.

## Index

| ADR | Decision | Summary |
|---|---|---|
| [0001](0001-dataset-is-not-committed.md) | The dataset is not committed | Download and SHA-256 verify the public source instead of committing it; test with synthetic fixtures. |
| [0002](0002-out-of-time-evaluation.md) | Out-of-time evaluation is primary | Reconstruct the calendar and make the chronological split the headline protocol; keep the random split for comparison. |
| [0003](0003-macro-features-are-a-calendar-proxy.md) | Macro variables are a calendar proxy | Report within-period metrics beside pooled ones and ship a `client_only` feature set rather than silently dropping the macro block. |
| [0004](0004-unknown-is-a-category-not-a-missing-value.md) | `unknown` is a category | Keep the literal source level instead of imputing it, and record the fairness caveat that comes with it. |
| [0005](0005-average-precision-and-validation-chosen-thresholds.md) | AP and validation-chosen thresholds | Lead with average precision against base rate, and select one threshold per model on validation only. |
| [0006](0006-experiment-tracking-behind-a-protocol.md) | Tracking behind a protocol | Depend on an `ExperimentTracker` protocol, default to JSONL, keep MLflow behind an optional extra. |
| [0007](0007-hand-rolled-table-schema-pydantic-at-the-edge.md) | Two boundary validators | A dependency-free table schema for training ingestion, Pydantic for record-level inference contracts. |
| [0008](0008-preprocessing-lives-inside-the-estimator.md) | Preprocessing inside the estimator | Return a fresh preprocessor per call and persist the fitted pipeline, not a bare estimator. |

All eight decisions were accepted on 2026-08-13.

## Format

Each record uses the same short structure:

- **Title** — the decision, stated as a fact rather than a question.
- **Status** — `Accepted`, plus the date. Later values would be `Superseded by NNNN` or `Deprecated`.
- **Context** — the forces at the time: the data, the constraint, the failure that prompted the choice.
- **Decision** — what was actually done, naming the modules and configuration keys involved.
- **Consequences** — both directions. An ADR with only positive consequences is not finished.
- **Alternatives considered** — the options rejected, and the specific reason each was rejected.

## Conventions

- Numbering is sequential and permanent. Numbers are never reused, even when a record is superseded.
- File names are `NNNN-kebab-case-title.md` and match the title.
- A new decision that changes an old one gets a new number and marks the old one `Superseded by NNNN`;
  the original text stays intact so the reasoning history survives.
- Every quantitative claim in an ADR is traceable to a generated file under `reports/metrics/` or
  `data/interim/`, or to code in `src/term_deposit/`. Nothing here is estimated from memory.
- ADRs describe this repository as it is. Nothing in it is deployed or running in production, and no
  ADR should be read as describing a live system.
