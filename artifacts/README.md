# Model artifacts

An artifact is the fitted scorer plus enough metadata to decide whether it is safe and compatible to
load. Artifacts are generated, version-sensitive outputs and are gitignored.

## Layout

```text
artifacts/
├── <run_id>/
│   ├── model.joblib
│   └── metadata.json
└── latest -> <run_id>
```

The run ID is `<UTC timestamp>__<protocol>__<feature_set>`, for example
`20260813T194016Z__out_of_time__all`. If a second run starts in the same second, `-2`, `-3`, and so on
are appended rather than overwriting the first.

`latest` is a directory symlink when supported. On filesystems without symlinks, persistence can write
a text pointer, and loading finally falls back to the most recently modified valid run directory.

## Files

`model.joblib` contains the fitted scoring object:

- column selection and ordering;
- `pdays` sentinel expansion;
- numeric scaling and categorical one-hot encoding;
- the selected estimator;
- the validation-fitted calibration map, when enabled.

`metadata.json` records:

| Field | Audit purpose |
|---|---|
| `schema_version` | Allows future readers to migrate the metadata format. |
| `model_name`, `estimator` | Identifies the candidate and implementation. |
| `split_strategy`, `feature_set` | States the evaluation protocol and input family. |
| `input_columns` | Defines the raw scoring contract and stable order. |
| `decision_threshold`, `threshold_objective` | Reproduces the shipped hard decision. |
| `metrics` | Stores test, within-period, backtest and CV evidence. |
| `created_at`, `git_revision` | Connects the binary to time and source revision. |
| `library_versions` | Warns when deserialisation or predictions may drift. |
| `config` | Captures every setting that shaped the run. |
| `data_checksum` | Identifies the exact source bytes. |
| `notes` | Carries limitations into inference output. |

The binary without metadata cannot explain its contract; metadata without the binary cannot score. A
run is valid only when both files exist.

## Loading

```python
from pathlib import Path

import pandas as pd

from term_deposit.inference import load_predictor

predictor = load_predictor(Path("artifacts"), model_id="latest")
customers = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
call_list = predictor.rank(customers, top_k=100)
```

`load_predictor()` validates required columns and values by default, applies the stored threshold, and
assigns rank-based priority tiers. Pin a specific run ID instead of `latest` for a reproducible campaign.

## Lifecycle

Create an artifact with `make train` or `make compare-protocols`. Re-evaluate it against the protocol it
records with `make evaluate`. Do not copy an artifact between library environments without checking
`library_versions`; joblib is not a language-neutral model format.

Artifacts are excluded from Git because they are regenerated from code, configuration and the
checksum-verified dataset. A real deployment would publish approved binaries to an access-controlled,
immutable registry with retention and rollback policies; this repository does not provide one.
