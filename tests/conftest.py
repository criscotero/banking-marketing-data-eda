"""Shared fixtures.

Tests run against a small synthetic dataset built here, not against the real
41k-row CSV. That keeps the suite fast, keeps CI independent of UCI's
availability, and lets each fixture encode exactly the property under test —
a chronological ordering, a known base-rate drift, a specific sentinel.

Tests that genuinely need the real file are marked ``requires_dataset`` and skip
when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from term_deposit import constants
from term_deposit.config import (
    AppConfig,
    CalibrationConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelSpec,
    PathsConfig,
    SplitConfig,
    TrackingConfig,
    TrainingConfig,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_csv_path() -> Path:
    """Path to the real dataset, if it has been downloaded."""
    return REPO_ROOT / "data" / "raw" / constants.RAW_FILENAME


@pytest.fixture(scope="session")
def require_dataset(raw_csv_path: Path) -> Path:
    """Skip the test when the real dataset is not present."""
    if not raw_csv_path.is_file():
        pytest.skip("real dataset not available; run scripts/prepare_data.py")
    return raw_csv_path


def _make_synthetic_frame(
    n_per_period: int = 240, n_periods: int = 14, seed: int = 7
) -> pd.DataFrame:
    """Build a dataset with the structure the pipeline cares about.

    Reproduces three properties of the real data on purpose:

    * rows ordered chronologically, with month rollovers,
    * macro indicators constant within each period,
    * a base rate that drifts upward across periods.

    A genuine within-customer signal is also injected (via ``age`` and
    ``poutcome``) so that a model can score above chance and tests can assert on
    ordering rather than on noise.
    """
    rng = np.random.default_rng(seed)
    months = list(constants.MONTH_ABBREVIATIONS)
    rows: list[dict[str, object]] = []

    for period_index in range(n_periods):
        month = months[(constants.CAMPAIGN_START_MONTH - 1 + period_index) % 12]
        euribor = 5.0 - 0.3 * period_index
        employed = 5200.0 - 15.0 * period_index
        base_logit = -2.5 + 0.22 * period_index  # drifting base rate

        for _ in range(n_per_period):
            age = int(rng.integers(18, 90))
            poutcome = str(rng.choice(["failure", "nonexistent", "success"], p=[0.1, 0.85, 0.05]))
            previously_contacted = poutcome != "nonexistent"
            logit = base_logit + 0.030 * (age - 40) + (1.4 if poutcome == "success" else 0.0)
            probability = 1.0 / (1.0 + np.exp(-logit))
            rows.append(
                {
                    "age": age,
                    "job": str(rng.choice(constants.CATEGORY_DOMAINS["job"])),
                    "marital": str(rng.choice(constants.CATEGORY_DOMAINS["marital"])),
                    "education": str(rng.choice(constants.CATEGORY_DOMAINS["education"])),
                    "default": str(rng.choice(["no", "unknown"], p=[0.8, 0.2])),
                    "housing": str(rng.choice(["no", "yes", "unknown"], p=[0.45, 0.5, 0.05])),
                    "loan": str(rng.choice(["no", "yes", "unknown"], p=[0.8, 0.15, 0.05])),
                    "contact": str(rng.choice(["cellular", "telephone"])),
                    "month": month,
                    "day_of_week": str(rng.choice(constants.CATEGORY_DOMAINS["day_of_week"])),
                    "duration": int(rng.integers(10, 900)),
                    "campaign": int(rng.integers(1, 6)),
                    "pdays": int(rng.integers(1, 30))
                    if previously_contacted
                    else constants.PDAYS_NEVER_CONTACTED,
                    "previous": int(previously_contacted),
                    "poutcome": poutcome,
                    "emp.var.rate": round(1.4 - 0.3 * period_index, 3),
                    "cons.price.idx": round(93.9 - 0.05 * period_index, 3),
                    "cons.conf.idx": round(-36.0 - 0.4 * period_index, 3),
                    "euribor3m": round(euribor, 3),
                    "nr.employed": round(employed, 1),
                    constants.TARGET_COLUMN: (
                        constants.TARGET_POSITIVE_LABEL
                        if rng.random() < probability
                        else constants.TARGET_NEGATIVE_LABEL
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """A synthetic dataset with the real file's raw column layout."""
    return _make_synthetic_frame()


@pytest.fixture
def raw_frame(synthetic_raw: pd.DataFrame) -> pd.DataFrame:
    """A fresh copy of the synthetic dataset for a single test."""
    return synthetic_raw.copy()


@pytest.fixture
def labelled_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Synthetic data with the binary label and reconstructed period attached."""
    from term_deposit.features.calendar import add_contact_period

    frame = raw_frame.copy()
    frame[constants.LABEL_COLUMN] = (
        frame[constants.TARGET_COLUMN] == constants.TARGET_POSITIVE_LABEL
    ).astype("int8")
    frame = frame.drop(columns=list(constants.POST_OUTCOME_COLUMNS))
    return add_contact_period(frame)


@pytest.fixture
def synthetic_csv(tmp_path: Path, raw_frame: pd.DataFrame) -> Path:
    """The synthetic dataset written out in the real file's CSV dialect."""
    path = tmp_path / constants.RAW_FILENAME
    raw_frame.to_csv(path, sep=constants.RAW_CSV_SEPARATOR, index=False)
    return path


@pytest.fixture
def paths(tmp_path: Path) -> PathsConfig:
    """A throwaway directory layout, so tests never touch the repository's."""
    config = PathsConfig(
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
    )
    config.ensure_directories()
    return config


@pytest.fixture
def fast_training_config() -> TrainingConfig:
    """Two cheap models, no cross-validation, no backtest."""
    return TrainingConfig(
        models=(
            ModelSpec(name="baseline_prior", estimator="dummy", params={"strategy": "prior"}),
            ModelSpec(
                name="logistic_regression",
                estimator="logistic_regression",
                params={"max_iter": 200},
                balance_strategy="class_weight",
            ),
        ),
        calibration=CalibrationConfig(enabled=False),
        cross_validate=False,
        backtest=False,
        n_jobs=1,
    )


@pytest.fixture
def app_config(paths: PathsConfig, fast_training_config: TrainingConfig) -> AppConfig:
    """A complete, fast configuration pointed at a temporary directory."""
    return AppConfig(
        paths=paths,
        split=SplitConfig(strategy="out_of_time", test_periods=3, validation_periods=3),
        features=FeatureConfig(),
        training=fast_training_config,
        evaluation=EvaluationConfig(n_bootstrap=0, within_period_min_rows=20),
        tracking=TrackingConfig(backend="none"),
        log_level="WARNING",
    )
