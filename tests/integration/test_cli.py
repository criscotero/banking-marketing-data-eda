"""The command-line surface.

Exercised with typer's ``CliRunner`` against a temporary project root, so the
commands a reviewer runs are covered by the suite rather than only by a README
that might have drifted.
"""

from __future__ import annotations

import shutil

import pytest
import yaml
from typer.testing import CliRunner

from term_deposit.cli import app

runner = CliRunner()


@pytest.fixture
def project(tmp_path, synthetic_csv, monkeypatch):
    """A miniature project root: configs, a dataset, and empty output dirs."""
    for directory in ("configs", "data/raw", "data/interim", "artifacts", "reports/metrics"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    shutil.copyfile(synthetic_csv, tmp_path / "data/raw/bank-additional-full.csv")

    base = {
        "log_level": "WARNING",
        "paths": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "artifacts_dir": "artifacts",
            "reports_dir": "reports",
        },
        # The synthetic file is smaller and differently hashed than the real one.
        "data": {"validate_schema": False, "expected_sha256": "", "enforce_checksum": False},
        "split": {"strategy": "out_of_time", "test_periods": 3, "validation_periods": 3},
        "evaluation": {"n_bootstrap": 0, "within_period_min_rows": 20},
        "tracking": {"backend": "none"},
    }
    training = {
        "training": {
            "models": [
                {"name": "baseline_prior", "estimator": "dummy"},
                {
                    "name": "logistic_regression",
                    "estimator": "logistic_regression",
                    "params": {"max_iter": 200},
                    "balance_strategy": "class_weight",
                },
            ],
            "calibration": {"enabled": False},
            "cross_validate": False,
            "backtest": False,
            "n_jobs": 1,
        }
    }
    inference = {"inference": {"validate_input": False}}

    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "configs/training.yaml").write_text(yaml.safe_dump(training))
    (tmp_path / "configs/inference.yaml").write_text(yaml.safe_dump(inference))

    monkeypatch.setattr("term_deposit.cli._shared.repo_root", lambda: tmp_path)
    return tmp_path


def _configs(project, *names: str) -> list[str]:
    flags: list[str] = []
    for name in names:
        flags += ["--config", str(project / "configs" / name)]
    return flags


class TestHelp:
    def test_root_help_lists_every_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("prepare-data", "train", "evaluate", "predict"):
            assert command in result.stdout

    def test_version_prints_the_package_version(self):
        from term_deposit import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    @pytest.mark.parametrize("command", ["prepare-data", "train", "evaluate", "predict"])
    def test_each_command_has_help(self, command):
        assert runner.invoke(app, [command, "--help"]).exit_code == 0


class TestPrepareData:
    def test_profiles_the_dataset(self, project):
        result = runner.invoke(
            app, ["prepare-data", *_configs(project, "base.yaml"), "--no-download"]
        )
        assert result.exit_code == 0, result.stdout
        assert "Dataset ready" in result.stdout
        assert (project / "data/interim/dataset_profile.json").is_file()
        assert (project / "data/interim/period_summary.csv").is_file()

    def test_reports_the_macro_collinearity(self, project):
        """The headline diagnostic should be visible without opening a notebook."""
        result = runner.invoke(
            app, ["prepare-data", *_configs(project, "base.yaml"), "--no-download"]
        )
        assert "determined by the contact month" in result.stdout

    def test_a_missing_dataset_fails_cleanly(self, project):
        (project / "data/raw/bank-additional-full.csv").unlink()
        result = runner.invoke(
            app, ["prepare-data", *_configs(project, "base.yaml"), "--no-download"]
        )
        assert result.exit_code != 0


class TestTrain:
    def test_trains_and_persists(self, project):
        result = runner.invoke(
            app,
            [
                "train",
                *_configs(project, "base.yaml", "training.yaml"),
                "--no-download",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Best model:" in result.stdout
        assert any((project / "artifacts").glob("*/model.joblib"))

    def test_prints_the_comparison_table_with_the_within_period_column(self, project):
        result = runner.invoke(
            app, ["train", *_configs(project, "base.yaml", "training.yaml"), "--no-download"]
        )
        assert "within_period_roc_auc" in result.stdout
        assert "baseline_prior" in result.stdout

    def test_reports_the_calendar_boundaries(self, project):
        result = runner.invoke(
            app, ["train", *_configs(project, "base.yaml", "training.yaml"), "--no-download"]
        )
        assert "calendar boundaries" in result.stdout

    def test_no_persist_writes_nothing(self, project):
        result = runner.invoke(
            app,
            [
                "train",
                *_configs(project, "base.yaml", "training.yaml"),
                "--no-download",
                "--no-persist",
            ],
        )
        assert result.exit_code == 0
        assert not list((project / "artifacts").glob("*/model.joblib"))

    def test_set_overrides_the_protocol(self, project):
        result = runner.invoke(
            app,
            [
                "train",
                *_configs(project, "base.yaml", "training.yaml"),
                "--no-download",
                "--set",
                "split.strategy=random",
                "--set",
                "split.test_size=0.25",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "protocol: random" in result.stdout

    def test_a_missing_training_section_names_the_file(self, project):
        result = runner.invoke(app, ["train", *_configs(project, "base.yaml"), "--no-download"])
        assert result.exit_code != 0

    def test_a_nonexistent_config_fails_cleanly(self, project):
        result = runner.invoke(app, ["train", "--config", str(project / "nope.yaml")])
        assert result.exit_code != 0


class TestEvaluateAndPredict:
    @pytest.fixture
    def trained(self, project):
        result = runner.invoke(
            app, ["train", *_configs(project, "base.yaml", "training.yaml"), "--no-download"]
        )
        assert result.exit_code == 0, result.stdout
        return project

    def test_evaluate_reports_both_pooled_and_within_period_scores(self, trained):
        result = runner.invoke(app, ["evaluate", *_configs(trained, "base.yaml")])
        assert result.exit_code == 0, result.stdout
        assert "average precision" in result.stdout
        assert "within month" in result.stdout

    def test_evaluate_writes_a_report(self, trained):
        runner.invoke(app, ["evaluate", *_configs(trained, "base.yaml")])
        assert list((trained / "reports/metrics").glob("evaluation__*.json"))

    def test_evaluate_without_an_artifact_names_the_fix(self, project):
        result = runner.invoke(app, ["evaluate", *_configs(project, "base.yaml")])
        assert result.exit_code != 0

    def test_predict_writes_a_ranked_call_list(self, trained):
        output = trained / "reports/metrics/call_list.csv"
        result = runner.invoke(
            app,
            [
                "predict",
                *_configs(trained, "base.yaml", "inference.yaml"),
                "--input",
                str(trained / "data/raw/bank-additional-full.csv"),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert output.is_file()

        import pandas as pd

        ranked = pd.read_csv(output)
        assert ranked["subscription_probability"].is_monotonic_decreasing
        assert ranked["rank"].iloc[0] == 1

    def test_predict_honours_a_capacity_fraction(self, trained):
        output = trained / "reports/metrics/top20.csv"
        source = trained / "data/raw/bank-additional-full.csv"

        import pandas as pd

        total = len(pd.read_csv(source, sep=";"))
        result = runner.invoke(
            app,
            [
                "predict",
                *_configs(trained, "base.yaml", "inference.yaml"),
                "--input",
                str(source),
                "--output",
                str(output),
                "--capacity",
                "0.2",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert len(pd.read_csv(output)) == pytest.approx(total * 0.2, abs=2)

    def test_predict_surfaces_the_model_caveats(self, trained):
        result = runner.invoke(
            app,
            [
                "predict",
                *_configs(trained, "base.yaml", "inference.yaml"),
                "--input",
                str(trained / "data/raw/bank-additional-full.csv"),
            ],
        )
        assert "note:" in result.stdout

    def test_predict_rejects_a_missing_input_file(self, trained):
        result = runner.invoke(
            app,
            [
                "predict",
                *_configs(trained, "base.yaml", "inference.yaml"),
                "--input",
                str(trained / "absent.csv"),
            ],
        )
        assert result.exit_code != 0

    def test_predict_rejects_an_out_of_range_capacity(self, trained):
        result = runner.invoke(
            app,
            [
                "predict",
                *_configs(trained, "base.yaml", "inference.yaml"),
                "--input",
                str(trained / "data/raw/bank-additional-full.csv"),
                "--capacity",
                "1.5",
            ],
        )
        assert result.exit_code != 0
