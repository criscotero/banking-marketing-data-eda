"""Raw-data contract and loading.

Schema validation is the project's only defence against a silently changed
upstream file. These tests check that it catches the realistic corruptions and
that it reports all of them at once rather than one per run.
"""

from __future__ import annotations

import pytest

from term_deposit import constants
from term_deposit.config import DataConfig
from term_deposit.data.loader import (
    ChecksumMismatchError,
    DatasetNotFoundError,
    ensure_raw_dataset,
    load_raw_dataset,
    sha256_of_file,
    write_sample,
)
from term_deposit.data.schema import (
    RAW_SCHEMA,
    ColumnSpec,
    SchemaValidationError,
    TableSchema,
    summarise_quality,
    validate_raw_dataframe,
)


class TestColumnSpec:
    def test_numeric_bounds_are_enforced(self):
        import pandas as pd

        spec = ColumnSpec("age", "numeric", minimum=17, maximum=120)
        assert spec.validate(pd.Series([20, 40])) == []
        assert "below minimum" in spec.validate(pd.Series([5]))[0]
        assert "above maximum" in spec.validate(pd.Series([900]))[0]

    def test_categorical_domains_are_enforced(self):
        import pandas as pd

        spec = ColumnSpec("marital", "categorical", allowed=("single", "married"))
        assert spec.validate(pd.Series(["single"])) == []
        assert "unexpected categories" in spec.validate(pd.Series(["widowed"]))[0]

    def test_nulls_are_rejected_in_non_nullable_columns(self):
        import pandas as pd

        spec = ColumnSpec("age", "numeric")
        assert "null value" in spec.validate(pd.Series([1.0, None]))[0]

    def test_a_wrong_dtype_is_reported_once(self):
        import pandas as pd

        spec = ColumnSpec("age", "numeric", minimum=0)
        problems = spec.validate(pd.Series(["forty", "fifty"]))
        assert len(problems) == 1
        assert "expected a numeric dtype" in problems[0]

    def test_unexpected_category_lists_are_truncated(self):
        import pandas as pd

        spec = ColumnSpec("job", "categorical", allowed=("a",))
        message = spec.validate(pd.Series([f"job{i}" for i in range(12)]))[0]
        assert "more)" in message


class TestValidateRawDataframe:
    def test_accepts_conforming_data(self, raw_frame):
        assert validate_raw_dataframe(raw_frame, check_row_count=False) is raw_frame

    def test_reports_every_violation_at_once(self, raw_frame):
        """Fixing an extract one error per run is the slowest possible loop."""
        broken = raw_frame.copy()
        broken.loc[0, "age"] = 500
        broken.loc[1, "marital"] = "widowed"
        broken.loc[2, "campaign"] = 0
        with pytest.raises(SchemaValidationError) as info:
            validate_raw_dataframe(broken, check_row_count=False)
        assert len(info.value.violations) >= 3

    def test_detects_a_missing_column(self, raw_frame):
        with pytest.raises(SchemaValidationError, match="missing required column"):
            validate_raw_dataframe(raw_frame.drop(columns=["euribor3m"]), check_row_count=False)

    def test_detects_an_empty_frame(self, raw_frame):
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_raw_dataframe(raw_frame.head(0), check_row_count=False)

    def test_row_count_is_only_checked_when_asked(self, raw_frame):
        with pytest.raises(SchemaValidationError, match="expected 41188 rows"):
            validate_raw_dataframe(raw_frame, check_row_count=True)
        validate_raw_dataframe(raw_frame, check_row_count=False)

    def test_rejects_an_unexpected_target_label(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[0, constants.TARGET_COLUMN] = "maybe"
        with pytest.raises(SchemaValidationError, match="unexpected categories"):
            validate_raw_dataframe(broken, check_row_count=False)

    def test_pdays_above_the_sentinel_is_rejected(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[0, "pdays"] = 1500
        with pytest.raises(SchemaValidationError, match="pdays"):
            validate_raw_dataframe(broken, check_row_count=False)


class TestTableSchema:
    def test_extra_columns_are_tolerated(self, raw_frame):
        """Derived columns are added downstream; the contract is a floor."""
        extended = raw_frame.assign(extra=1)
        assert RAW_SCHEMA.validate(extended, check_row_count=False) == []

    def test_column_names_lists_every_spec(self):
        assert len(RAW_SCHEMA.column_names) == 21
        assert constants.TARGET_COLUMN in RAW_SCHEMA.column_names

    def test_a_custom_schema_can_require_a_subset(self, raw_frame):
        schema = TableSchema(columns=(ColumnSpec("age", "numeric", minimum=0),), required=("age",))
        assert schema.validate(raw_frame[["age"]], check_row_count=False) == []


class TestSummariseQuality:
    def test_counts_the_literal_unknown_category_separately_from_nulls(self, raw_frame):
        """``unknown`` is a recorded non-answer, not a missing value.

        Conflating the two would erase the distinction the project deliberately
        preserves by declining to impute it.
        """
        summary = summarise_quality(raw_frame).set_index("column")
        assert int(summary["n_unknown"]["default"]) > 0
        assert int(summary["n_missing"]["default"]) == 0

    def test_covers_every_column(self, raw_frame):
        assert len(summarise_quality(raw_frame)) == raw_frame.shape[1]


class TestLoadRawDataset:
    def test_reads_the_semicolon_dialect_and_adds_the_binary_label(self, synthetic_csv):
        frame = load_raw_dataset(synthetic_csv, check_row_count=False)
        assert constants.LABEL_COLUMN in frame.columns
        assert set(frame[constants.LABEL_COLUMN].unique()) <= {0, 1}

    def test_the_label_matches_the_original_target(self, synthetic_csv):
        frame = load_raw_dataset(synthetic_csv, check_row_count=False)
        expected = (frame[constants.TARGET_COLUMN] == constants.TARGET_POSITIVE_LABEL).astype(int)
        assert (frame[constants.LABEL_COLUMN] == expected).all()

    def test_preserves_row_order(self, synthetic_csv, raw_frame):
        """Row order carries the contact chronology; reordering would destroy it."""
        frame = load_raw_dataset(synthetic_csv, check_row_count=False)
        assert frame["month"].tolist() == raw_frame["month"].tolist()

    def test_a_missing_file_names_the_fix(self, tmp_path):
        with pytest.raises(DatasetNotFoundError, match=r"prepare_data.py"):
            load_raw_dataset(tmp_path / "absent.csv")

    def test_validation_can_be_switched_off(self, tmp_path, raw_frame):
        broken = raw_frame.copy()
        broken.loc[0, "age"] = 900
        path = tmp_path / "broken.csv"
        broken.to_csv(path, sep=constants.RAW_CSV_SEPARATOR, index=False)
        assert len(load_raw_dataset(path, validate=False)) == len(broken)


class TestChecksums:
    def test_digest_is_stable_and_content_sensitive(self, tmp_path):
        first = tmp_path / "a.txt"
        first.write_text("hello")
        second = tmp_path / "b.txt"
        second.write_text("hello")
        third = tmp_path / "c.txt"
        third.write_text("hello!")
        assert sha256_of_file(first) == sha256_of_file(second)
        assert sha256_of_file(first) != sha256_of_file(third)

    def test_ensure_raw_dataset_accepts_a_matching_file(self, synthetic_csv):
        config = DataConfig(expected_sha256=sha256_of_file(synthetic_csv))
        assert ensure_raw_dataset(config, synthetic_csv, download=False) == synthetic_csv

    def test_ensure_raw_dataset_rejects_a_mismatch(self, synthetic_csv):
        """A silently changed upstream file must not become a reported metric."""
        config = DataConfig(expected_sha256="0" * 64)
        with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
            ensure_raw_dataset(config, synthetic_csv, download=False)

    def test_enforce_checksum_false_downgrades_to_a_warning(self, synthetic_csv):
        config = DataConfig(expected_sha256="0" * 64, enforce_checksum=False)
        assert ensure_raw_dataset(config, synthetic_csv, download=False) == synthetic_csv

    def test_a_missing_file_without_download_names_the_fix(self, tmp_path):
        with pytest.raises(DatasetNotFoundError, match=r"prepare_data.py"):
            ensure_raw_dataset(DataConfig(), tmp_path / "absent.csv", download=False)


class TestWriteSample:
    def test_writes_the_header_plus_n_rows(self, synthetic_csv, tmp_path):
        sample = write_sample(synthetic_csv, tmp_path / "sample.csv", n_rows=50)
        assert len(sample.read_text().splitlines()) == 51

    def test_the_sample_preserves_chronological_order(self, synthetic_csv, tmp_path):
        """Sampling randomly would break the calendar reconstruction."""
        import pandas as pd

        sample = write_sample(synthetic_csv, tmp_path / "sample.csv", n_rows=100)
        frame = pd.read_csv(sample, sep=constants.RAW_CSV_SEPARATOR)
        original = pd.read_csv(synthetic_csv, sep=constants.RAW_CSV_SEPARATOR).head(100)
        assert frame["month"].tolist() == original["month"].tolist()
