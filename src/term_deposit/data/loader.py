"""Fetching and reading the raw dataset.

The dataset is not committed to the repository. `download_raw_dataset` retrieves
it from UCI and verifies a SHA-256 checksum, so a reported metric can always be
traced back to a specific bag of bytes. See ADR 0001.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from term_deposit import constants
from term_deposit.config import DataConfig
from term_deposit.data.schema import validate_raw_dataframe
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 120
_HASH_CHUNK_BYTES = 1 << 20


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw CSV is absent and no download was requested."""


class ChecksumMismatchError(RuntimeError):
    """Raised when the downloaded file does not match the expected checksum."""


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_dataset(
    destination: Path,
    *,
    source_url: str = constants.UCI_ARCHIVE_URL,
    expected_sha256: str | None = constants.RAW_SHA256,
    force: bool = False,
) -> Path:
    """Download ``bank-additional-full.csv`` from UCI into ``destination``.

    The published archive nests a second zip, so both layers are unpacked in
    memory and only the single CSV member is written to disk.

    Args:
        destination: Target CSV path. Parent directories are created.
        source_url: Outer archive URL.
        expected_sha256: Digest the extracted CSV must match. ``None`` skips the check.
        force: Re-download even when a valid file is already present.

    Returns:
        The path to the verified CSV.

    Raises:
        ChecksumMismatchError: If the extracted file's digest differs.
        RuntimeError: If the download fails or the archive lacks the expected member.
    """
    if destination.exists() and not force:
        if expected_sha256 is None or sha256_of_file(destination) == expected_sha256:
            logger.info("Raw dataset already present at %s", destination)
            return destination
        logger.warning("Existing file at %s failed its checksum; re-downloading", destination)

    logger.info("Downloading %s", source_url)
    try:
        with urllib.request.urlopen(source_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        msg = (
            f"could not download the dataset from {source_url}: {error}. "
            f"Download it manually and place it at {destination}."
        )
        raise RuntimeError(msg) from error

    csv_bytes = _extract_csv_bytes(payload, source_url)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)

    if expected_sha256 is not None:
        actual = sha256_of_file(destination)
        if actual != expected_sha256:
            destination.unlink(missing_ok=True)
            msg = (
                f"checksum mismatch for the downloaded dataset: "
                f"expected {expected_sha256}, got {actual}. "
                "The upstream file may have changed; do not report metrics from it."
            )
            raise ChecksumMismatchError(msg)

    logger.info("Wrote verified dataset to %s (%d bytes)", destination, len(csv_bytes))
    return destination


def _extract_csv_bytes(archive_bytes: bytes, source_url: str) -> bytes:
    """Pull the CSV out of the (doubly nested) UCI archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
            names = outer.namelist()
            if constants.UCI_INNER_ARCHIVE in names:
                inner_bytes = outer.read(constants.UCI_INNER_ARCHIVE)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    return inner.read(constants.UCI_INNER_MEMBER)
            direct = next(
                (name for name in names if name.endswith(constants.RAW_FILENAME)),
                None,
            )
            if direct is not None:
                return outer.read(direct)
    except (zipfile.BadZipFile, KeyError) as error:
        msg = f"unexpected archive layout at {source_url}: {error}"
        raise RuntimeError(msg) from error

    msg = f"{constants.RAW_FILENAME} not found inside the archive at {source_url}"
    raise RuntimeError(msg)


def ensure_raw_dataset(config: DataConfig, path: Path, *, download: bool = True) -> Path:
    """Guarantee that a checksum-verified raw CSV exists at ``path``.

    Args:
        config: Data settings (source URL, expected digest, enforcement flag).
        path: Where the CSV should live.
        download: Fetch the file when it is missing. ``False`` turns a missing
            file into an error, which is what CI wants.

    Returns:
        The path to the CSV.

    Raises:
        DatasetNotFoundError: If the file is missing and ``download`` is false.
        ChecksumMismatchError: If the file is present but does not match and
            ``config.enforce_checksum`` is true.
    """
    if not path.exists():
        if not download:
            msg = (
                f"raw dataset not found at {path}. "
                "Run `uv run python scripts/prepare_data.py` to download it."
            )
            raise DatasetNotFoundError(msg)
        return download_raw_dataset(
            path,
            source_url=config.source_url,
            expected_sha256=config.expected_sha256 if config.enforce_checksum else None,
        )

    if config.expected_sha256:
        actual = sha256_of_file(path)
        if actual != config.expected_sha256:
            message = (
                f"checksum mismatch for {path}: expected {config.expected_sha256}, got {actual}"
            )
            if config.enforce_checksum:
                raise ChecksumMismatchError(message)
            logger.warning("%s (enforce_checksum is off)", message)
    return path


def load_raw_dataset(
    path: Path,
    *,
    validate: bool = True,
    check_row_count: bool = True,
) -> pd.DataFrame:
    """Read the raw semicolon-delimited CSV and add the binary label.

    Args:
        path: CSV location.
        validate: Enforce :data:`~term_deposit.data.schema.RAW_SCHEMA`.
        check_row_count: Include the exact row count in validation. Off for samples.

    Returns:
        The raw columns plus ``subscribed`` (``0``/``1``). The original ``y``
        column is retained so that the frame still satisfies the raw contract.

    Raises:
        DatasetNotFoundError: If ``path`` does not exist.
        SchemaValidationError: If validation is on and the contract is violated.
    """
    if not path.exists():
        msg = (
            f"raw dataset not found at {path}. "
            "Run `uv run python scripts/prepare_data.py` to download it."
        )
        raise DatasetNotFoundError(msg)

    frame = pd.read_csv(path, sep=constants.RAW_CSV_SEPARATOR)
    logger.info("Loaded %d rows x %d columns from %s", len(frame), frame.shape[1], path)

    if validate:
        validate_raw_dataframe(frame, check_row_count=check_row_count)

    frame[constants.LABEL_COLUMN] = (
        frame[constants.TARGET_COLUMN] == constants.TARGET_POSITIVE_LABEL
    ).astype("int8")
    return frame


def write_sample(source: Path, destination: Path, *, n_rows: int = 2000) -> Path:
    """Write the first ``n_rows`` of the raw CSV, preserving the header.

    Used to build the committed test fixture: it keeps the file's chronological
    ordering intact, which the calendar reconstruction depends on.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for index, line in enumerate(reader):
            if index > n_rows:
                break
            writer.write(line)
    return destination


def copy_local_dataset(source: Path, destination: Path) -> Path:
    """Copy an already-downloaded CSV into the project's raw directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    logger.info("Copied %s -> %s", source, destination)
    return destination
