# ADR 0006: Put experiment tracking behind a protocol

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

Every training run needs its parameters, metrics and artifact locations recorded, otherwise a reported
number cannot be traced back to the configuration that produced it.

The obvious tool is MLflow. But a portfolio repository should stay runnable from a fresh clone with no
tracking server, no credentials and no heavy optional SDK. Making MLflow mandatory would mean a reader
who only wants to run `make train` first has to stand up infrastructure that has nothing to do with the
modelling question.

At the same time, the experiment pipeline needs very little from a tracker: start a run, log
parameters, log scalar metrics, log an artifact. Importing a product API directly for those four calls
would push backend concerns into training code and make the pipeline hard to test.

## Decision

Declare `ExperimentTracker` in `src/term_deposit/tracking.py` as a `typing.Protocol` (structural,
`@runtime_checkable`) with exactly those four operations. The experiment pipeline depends on the
protocol and never on a concrete backend.

Ship three implementations in the same module:

- `JsonlTracker` — the default. Appends one line per run to `reports/metrics/runs.jsonl` and writes a
  full record to `reports/metrics/runs/<run_id>.json`. No dependencies beyond the standard library.
- `MlflowTracker` — optional, selected by `tracking.backend: mlflow`. It imports `mlflow` lazily and
  raises a message pointing at `uv sync --extra mlflow` when the extra is not installed.
- `NullTracker` — a no-op backend for tests and for runs where recording is not wanted.

`build_tracker(config, output_dir)` constructs the selected implementation from configuration.

## Consequences

Positive:

- A fresh clone records runs with no service, credentials or network.
- JSONL is append-only and readable with `jq`, pandas or any text tool; no schema migration needed.
- MLflow stays out of the default dependency surface and out of the default install.
- Tests substitute `NullTracker` without patching or a running server.
- Swapping backends is a configuration change, not a code change.

Negative:

- JSONL has no concurrent-write coordination; parallel runs writing the same file can interleave.
- There is no query service, comparison UI or artifact browser in the default path.
- Run records accumulate locally and are not archived or garbage-collected by anything here.
- Backend parity is limited to the four protocol methods; MLflow's model registry, nested runs and
  autologging are not exposed and would need code outside this interface.

## Alternatives considered

- **Mandatory MLflow:** rejected. The setup, dependency weight and credential story cost more than the
  project needs, and it would make the repository harder to evaluate quickly.
- **No tracking at all:** rejected. Metrics without the parameters, config and revision that produced
  them are not auditable, which defeats the purpose of the reports.
- **Write only the comparison CSV:** rejected. It captures results but omits configuration, tags and
  artifact paths, so a run cannot be reconstructed from it.
- **A small database (SQLite or similar):** rejected as infrastructure without a demonstrated need;
  there is no concurrency or query requirement that JSONL fails to meet at this scale.
- **Weights & Biases or a hosted equivalent:** rejected for the same reason as mandatory MLflow, with
  the additional problem of requiring an account to reproduce results.
