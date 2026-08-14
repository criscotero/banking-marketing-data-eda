"""Logging setup.

One `configure_logging` call at the entry point; every module then uses
`get_logger(__name__)`. Library code never configures handlers and never prints,
so importing the package from a notebook or a test leaves the host's logging
configuration untouched.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_ROOT_LOGGER_NAME = "term_deposit"


def configure_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger once and return it.

    Args:
        level: Level name or numeric level. ``TERM_DEPOSIT_LOG_LEVEL`` wins if set.
        log_file: Optional file to mirror records into; parent dirs are created.
        force: Replace existing handlers instead of leaving them in place.

    Returns:
        The configured ``term_deposit`` logger.
    """
    resolved = os.environ.get("TERM_DEPOSIT_LOG_LEVEL", level)
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(resolved)
    # Records are emitted by this logger's handlers only; the root logger's
    # configuration (e.g. a notebook's) must not duplicate them.
    logger.propagate = False

    if logger.handlers and not force:
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``term_deposit``.

    Args:
        name: Usually ``__name__``. The ``term_deposit.`` prefix is added if absent
            so that third-party callers still land under the package's logger.
    """
    if not name or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
