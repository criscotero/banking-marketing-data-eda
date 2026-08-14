# Multi-stage build. The runtime image carries the runtime dependencies and the
# package — no compilers, no dev tools, no test suite — so what ships is what
# scores. The `dev` stage adds the tooling for running the suite in CI.

# --------------------------------------------------------------------------- #
# Builder: resolve and install dependencies against the lockfile
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed before the source is copied, so editing a module
# does not invalidate the (slow) dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

# libgomp is required by the XGBoost and scikit-learn wheels.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root: an image that trains on downloaded data should not be able
# to write outside its own working directory.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app pyproject.toml README.md ./

RUN mkdir -p data/raw data/interim data/processed artifacts reports/figures reports/metrics \
    && chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERM_DEPOSIT_ROOT=/app

USER app

# Fails while the package cannot be imported or the config cannot be parsed —
# the two things that make every command in the image useless.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from term_deposit.config import load_config; load_config(['configs/base.yaml'])" || exit 1

ENTRYPOINT ["term-deposit"]
CMD ["--help"]

# --------------------------------------------------------------------------- #
# Dev: adds the dev dependency group and the test suite
# --------------------------------------------------------------------------- #
FROM runtime AS dev

USER root
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen && chown -R app:app /app/.venv

USER app
ENTRYPOINT []
CMD ["pytest", "-q"]
