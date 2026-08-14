"""Shared plumbing for the command implementations.

Configuration resolution and error reporting are identical across commands, so
they live here rather than being copy-pasted four times.
"""

from __future__ import annotations

import sys
from pathlib import Path

from term_deposit.config import AppConfig, ConfigError, load_config
from term_deposit.utils.logging import configure_logging

DEFAULT_BASE_CONFIG = Path("configs/base.yaml")


def repo_root() -> Path:
    """Repository root, inferred from this file's location.

    ``src/term_deposit/cli/_shared.py`` -> ``src/term_deposit/cli`` ->
    ``src/term_deposit`` -> ``src`` -> root.
    """
    return Path(__file__).resolve().parents[3]


def resolve_config(
    config_paths: list[Path] | None,
    overrides: list[str] | None,
    *,
    extra_defaults: tuple[Path, ...] = (),
) -> AppConfig:
    """Load configuration and configure logging.

    Args:
        config_paths: Explicit ``--config`` values, merged left to right. When
            given, they replace the defaults entirely, so a caller can run
            against a config set that has nothing to do with ``configs/``.
        overrides: ``--set key=value`` strings applied after the files.
        extra_defaults: Files layered after ``base.yaml`` when no explicit
            ``--config`` was given — this is how ``train`` picks up
            ``training.yaml`` without the caller naming it.

    Returns:
        A resolved :class:`~term_deposit.config.AppConfig` with absolute paths.

    Raises:
        ConfigError: If a file is missing or the merged config fails validation.
    """
    root = repo_root()
    if config_paths:
        candidates = [Path(path) for path in config_paths]
    else:
        candidates = [DEFAULT_BASE_CONFIG, *extra_defaults]

    resolved: list[Path] = []
    for path in candidates:
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            resolved.append(absolute)
        elif config_paths:
            # An explicitly named file that does not exist is an error; a missing
            # default is simply skipped.
            msg = f"config file not found: {path}"
            raise ConfigError(msg)

    if not resolved:
        msg = (
            f"no configuration found. Expected {DEFAULT_BASE_CONFIG} relative to {root}; "
            "pass --config to point somewhere else."
        )
        raise ConfigError(msg)

    config = load_config(resolved, overrides=overrides or [], project_root=root)
    configure_logging(config.log_level, force=True)
    return config


def fail(message: str, *, code: int = 1) -> None:
    """Print an actionable error to stderr and exit.

    Used for expected failures — a missing dataset, an invalid config, an unknown
    artifact — so the user sees one clear line instead of a traceback.
    """
    print(f"error: {message}", file=sys.stderr)  # noqa: T201 - user-facing CLI error
    raise SystemExit(code)


__all__ = ["DEFAULT_BASE_CONFIG", "ConfigError", "fail", "repo_root", "resolve_config"]
