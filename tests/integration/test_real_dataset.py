"""End-to-end checks against the real UCI dataset.

Every other test runs on synthetic fixtures so that CI needs no network. These
run on the actual file and exist to pin the claims the documentation makes. If
the upstream data changed, or a refactor quietly altered the protocol, the
README would start asserting numbers the code no longer produces — these tests
fail first.

They are marked ``slow`` and ``requires_dataset`` and are skipped unless
``data/raw/bank-additional-full.csv`` is present. Run them with::

    uv run pytest -m requires_dataset
"""

from __future__ import annotations

from pathlib import Path

import pytest

from term_deposit import constants
from term_deposit.config import load_config
from term_deposit.data.loader import load_raw_dataset
from term_deposit.features.calendar import (
    add_contact_period,
    macro_period_collinearity,
    period_summary,
)
from term_deposit.pipelines.experiment import (
    ExperimentResult,
    prepare_dataset,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [REPO_ROOT / "configs" / "base.yaml", REPO_ROOT / "configs" / "training.yaml"]

pytestmark = pytest.mark.requires_dataset


@pytest.fixture(scope="module")
def real_frame(require_dataset: Path):
    """The real dataset with the label and reconstructed calendar attached."""
    frame = load_raw_dataset(require_dataset)
    return add_contact_period(frame)


class TestDatasetShape:
    """The published extent of the file, pinned so a change is loud."""

    def test_row_count_and_base_rate(self, real_frame):
        assert len(real_frame) == constants.RAW_ROW_COUNT
        assert float(real_frame[constants.LABEL_COLUMN].mean()) == pytest.approx(0.1127, abs=0.0001)

    def test_campaign_window(self, real_frame):
        periods = real_frame[constants.PERIOD_COLUMN]
        assert str(periods.min()) == "2008-05"
        assert str(periods.max()) == "2010-11"
        assert periods.nunique() == 26


class TestTheCentralClaim:
    """The finding the whole project rests on, asserted against the real file."""

    def test_macro_features_are_determined_by_the_contact_month(self, real_frame):
        """Four macro features are exactly constant within a calendar month.

        This is the premise of the leakage argument. If it stopped holding, the
        README's explanation of *why* the random split inflates the score would
        be wrong even if the numbers still looked similar.
        """
        collinearity = macro_period_collinearity(real_frame).set_index("feature")

        for feature in ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed"):
            share = float(collinearity["between_period_variance_share"][feature])
            assert share == pytest.approx(1.0, abs=1e-9), f"{feature} is no longer period-constant"

        # euribor3m is a daily rate, so it moves slightly within a month.
        assert float(collinearity["between_period_variance_share"]["euribor3m"]) > 0.999

    def test_the_base_rate_drifts_across_the_campaign(self, real_frame):
        """The other half of the leakage mechanism.

        The documentation quotes the drift as 3.1% in 2008-05 to 57.5% in
        2010-05, naming both months. That is deliberate: the arithmetic maximum
        is 62.7%, but it belongs to 2008-10, a month with only 67 contacts.
        Quoting it as "the peak" would rest the argument on a rounding artefact,
        so both the named-month claim and the true extremes are pinned here.
        """
        summary = period_summary(real_frame)
        summary.index = summary["contact_period"].astype(str)

        assert float(summary["subscription_rate"]["2008-05"]) == pytest.approx(0.0309, abs=0.001)
        assert float(summary["subscription_rate"]["2010-05"]) == pytest.approx(0.5755, abs=0.001)

        # The true extremes, and the caveat that makes the maximum unquotable.
        rates = summary["subscription_rate"]
        assert float(rates.min()) == pytest.approx(0.0309, abs=0.001)
        assert float(rates.max()) == pytest.approx(0.6269, abs=0.001)
        assert int(summary["n_contacts"][rates.idxmax()]) == 67

    def test_the_early_months_dominate_the_row_count(self, real_frame):
        """The early months dominate the row count.

        Most rows come from the low-conversion 2008 window, which is why a
        pooled metric is a poor summary of recent performance.
        """
        summary = period_summary(real_frame)
        assert float(summary["share_of_rows"].head(4).sum()) > 0.55


class TestPreparedDataset:
    def test_the_post_outcome_column_is_dropped(self):
        config = load_config([CONFIGS[0]], project_root=REPO_ROOT)
        prepared = prepare_dataset(config, download=False)
        assert "duration" not in prepared.frame.columns
        assert prepared.checksum == constants.RAW_SHA256

    def test_the_feature_set_is_nineteen_columns(self):
        config = load_config([CONFIGS[0]], project_root=REPO_ROOT)
        assert len(config.features.input_columns()) == 19


@pytest.mark.slow
class TestProtocolGap:
    """Train real models and assert the documented protocol gap.

    This is the expensive one: it fits the configured models on the full dataset.
    Cross-validation and the backtest are disabled because this test is about the
    split, not about model selection.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def random_split_result(cls) -> ExperimentResult:
        """Train every configured model once, shared by the assertions below."""
        config = load_config(
            CONFIGS,
            overrides=[
                "split.strategy=random",
                "training.cross_validate=false",
                "training.backtest=false",
                "evaluation.n_bootstrap=0",
                "tracking.backend=none",
            ],
            project_root=REPO_ROOT,
        )
        return run_experiment(config, persist=False, download=False)

    def test_the_random_split_reproduces_the_familiar_roc_auc(self, random_split_result):
        """Around 0.81, as published for this dataset.

        This is the starting point of the argument, not its conclusion.
        """
        by_name = {r.model_name: r for r in random_split_result.reports}
        assert by_name["random_forest"].test_metrics.roc_auc == pytest.approx(0.81, abs=0.02)

    def test_pooled_roc_auc_is_substantially_inflated(self, random_split_result):
        """The headline claim.

        Every real model should show a large positive gap between its pooled and
        its within-month ROC-AUC.
        """
        for report in random_split_result.reports:
            if report.model_name == "baseline_prior":
                continue
            inflation = report.within_period["roc_auc_inflation"]
            assert inflation > 0.15, (
                f"{report.model_name} shows an inflation of {inflation:.4f}; "
                "the documentation claims roughly +0.22 for every model"
            )

    def test_within_month_ranking_is_close_to_chance(self, random_split_result):
        by_name = {r.model_name: r for r in random_split_result.reports}
        within = float(by_name["random_forest"].within_period["weighted_roc_auc"])
        assert 0.55 < within < 0.65

    def test_the_baseline_shows_no_inflation(self, random_split_result):
        """A constant score cannot separate months, so its gap must be zero.

        Anything else would mean the within-period metric itself is leaking.
        """
        by_name = {r.model_name: r for r in random_split_result.reports}
        baseline = by_name["baseline_prior"]
        assert baseline.within_period["roc_auc_inflation"] == pytest.approx(0.0, abs=1e-9)
        assert baseline.test_metrics.top_k["0.20"]["lift"] == pytest.approx(1.0, abs=0.05)
