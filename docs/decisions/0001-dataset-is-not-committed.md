# ADR 0001: The dataset is not committed

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

Bank Marketing is publicly downloadable from UCI under CC BY 4.0. Committing the nested archive or
extracted CSV would duplicate a source of record, add generated data to repository history, and make a
data change difficult to distinguish from a code change. The source zip is small (about 381 KB), so
specialised large-file infrastructure would cost more than it saves.

Reported metrics still need byte-level provenance. Depending on a mutable URL without verification
would allow an upstream replacement to change results under the same commit. Pull-request CI should
also remain reliable when UCI or the network is unavailable.

## Decision

Do not commit files under `data/raw/`, `data/interim/`, or `data/processed/` except `.gitkeep` and
documentation. `scripts/prepare_data.py` downloads the UCI archive, extracts only
`bank-additional-full.csv`, and verifies SHA-256:

```text
74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
```

The default test suite uses synthetic fixtures. Real-dataset tests carry the `requires_dataset` marker,
and the network-dependent end-to-end CI job is non-blocking and does not run on pull requests.

## Consequences

Positive:

- Git history contains code and contracts rather than reproducible data products.
- A metric can be tied to an exact extracted file through its checksum.
- Pull requests do not fail because a third-party download is unavailable.
- Contributors can replace the source deliberately through config and receive a checksum failure.

Negative:

- A first full run needs network access or a manually supplied local copy.
- UCI archive layout is an external dependency of the downloader.
- Reviewers cannot run real-data commands entirely offline from a fresh clone.

## Alternatives considered

- **Commit the CSV or zip:** simplest offline clone, rejected because it duplicates generated data in
  permanent history.
- **Git LFS:** useful for large binaries, unnecessary for a file of this size and adds service setup.
- **DVC:** useful for versioned multi-dataset pipelines, unnecessary for one immutable public input.
- **Trust the URL without a checksum:** less code, rejected because results would lose byte provenance.
