"""Pre-call propensity scoring for bank term-deposit telemarketing campaigns.

The package is organised as a one-way dependency chain so that any stage can be
imported, tested and reused on its own::

    data -> features -> models -> training -> evaluation -> inference

`pipelines` wires those stages into end-to-end experiments and `scripts/` (plus
the `term-deposit` console entry point) are the only user-facing surfaces.
"""

from __future__ import annotations

from term_deposit.config import AppConfig, load_config

__all__ = ["AppConfig", "__version__", "load_config"]

__version__ = "1.0.0"
