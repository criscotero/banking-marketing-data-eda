"""Deterministic seeding.

Two identical runs of `scripts/train.py` must produce byte-identical metrics.
That requires seeding `random` and `numpy` *and* pinning the thread counts of the
BLAS backends, because reductions over floats are order-dependent and thread
scheduling is not deterministic.
"""

from __future__ import annotations

import os
import random

import numpy as np

from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variables that force single-threaded BLAS. Set before the first
#: numpy-backed computation, they remove the last source of run-to-run drift.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def seed_everything(seed: int, *, deterministic_threads: bool = False) -> int:
    """Seed every source of randomness the pipeline touches.

    Args:
        seed: The seed to apply to ``random`` and ``numpy.random``.
        deterministic_threads: Also pin BLAS thread counts to 1. This makes
            float reductions bit-reproducible at a real cost in speed, so it is
            opt-in and reserved for reproducibility checks.

    Returns:
        The seed, for convenient chaining into estimator constructors.

    Note:
        Estimators receive the seed explicitly through their config; the global
        seeds here cover code that reaches for the module-level RNG (SHAP
        background sampling, scikit-learn internals without a ``random_state``).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG used by SHAP
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_threads:
        for variable in _THREAD_ENV_VARS:
            os.environ[variable] = "1"
        logger.debug("Pinned BLAS thread counts to 1 for bit-reproducibility")

    logger.debug("Seeded RNGs with %d", seed)
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """Return an explicit generator.

    Preferred over the global RNG anywhere the package draws random numbers, so
    that sampling is reproducible regardless of what else ran first.
    """
    return np.random.default_rng(seed)
