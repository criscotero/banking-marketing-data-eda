#!/usr/bin/env python
r"""Score customers and write a prioritised call list.

    uv run python scripts/predict.py --input data/raw/bank-additional-full.csv

The implementation lives in :mod:`term_deposit.cli.predict` so that it can be
imported and tested; this file is the documented entry point. The same command is
available as ``term-deposit predict`` once the package is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support `python scripts/predict.py` from a checkout where the project has not
# been installed. `uv run` installs it, so this is a no-op in the documented path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from term_deposit.cli.predict import app  # noqa: E402

if __name__ == "__main__":
    app()
