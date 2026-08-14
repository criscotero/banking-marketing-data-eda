# Every command a reviewer or a CI job needs, in one place.
# `make help` lists them; `make all` runs the full path from a clean checkout.

.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
RUN := $(UV) run
CONFIG := --config configs/base.yaml
TRAIN_CONFIG := $(CONFIG) --config configs/training.yaml

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

.PHONY: install
install: ## Create the virtualenv and install all dependencies
	$(UV) sync --all-extras

.PHONY: install-min
install-min: ## Install runtime + dev only (no viz, explain, mlflow, notebooks)
	$(UV) sync

.PHONY: hooks
hooks: ## Install the pre-commit hooks
	$(RUN) pre-commit install

# --------------------------------------------------------------------------- #
# Quality gates — the same set CI runs
# --------------------------------------------------------------------------- #

.PHONY: lint
lint: ## Lint with ruff
	$(RUN) ruff check .

.PHONY: format
format: ## Format with ruff
	$(RUN) ruff format .

.PHONY: format-check
format-check: ## Verify formatting without writing
	$(RUN) ruff format --check .

.PHONY: typecheck
typecheck: ## Static type check with mypy
	$(RUN) mypy

.PHONY: test
test: ## Run the test suite
	$(RUN) pytest

.PHONY: test-fast
test-fast: ## Run only tests that need no dataset
	$(RUN) pytest -m "not slow and not requires_dataset"

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(RUN) pytest --cov --cov-report=term-missing --cov-report=xml

.PHONY: check
check: lint format-check typecheck test ## Run every quality gate

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

.PHONY: data
data: ## Download, verify and profile the dataset
	$(RUN) python scripts/prepare_data.py $(CONFIG)

.PHONY: train
train: ## Train under the out-of-time protocol (the default)
	$(RUN) python scripts/train.py $(TRAIN_CONFIG)

.PHONY: train-random
train-random: ## Train under the random protocol, for comparison
	$(RUN) python scripts/train.py $(TRAIN_CONFIG) --set split.strategy=random

.PHONY: train-client-only
train-client-only: ## Train without the macro block, to isolate its contribution
	$(RUN) python scripts/train.py $(TRAIN_CONFIG) --set features.feature_set=client_only

.PHONY: compare-protocols
compare-protocols: ## Run both protocols and print the contrast that drives the README
	$(MAKE) train-random
	$(MAKE) train
	@echo ""
	@echo "Comparison tables are in reports/metrics/*__comparison.csv"

.PHONY: evaluate
evaluate: ## Re-evaluate the saved model and render the figures
	$(RUN) python scripts/evaluate.py $(CONFIG) --figures

.PHONY: predict
predict: ## Score the dataset into a ranked call list
	$(RUN) python scripts/predict.py $(CONFIG) --config configs/inference.yaml \
		--input data/raw/bank-additional-full.csv \
		--output reports/metrics/call_list.csv

.PHONY: all
all: install data train evaluate ## Full path from a clean checkout to results

# --------------------------------------------------------------------------- #
# Notebooks
# --------------------------------------------------------------------------- #

.PHONY: notebooks
notebooks: ## Launch JupyterLab
	$(RUN) jupyter lab notebooks/

.PHONY: notebooks-check
notebooks-check: ## Execute every notebook top to bottom and fail on any error
	$(RUN) jupyter nbconvert --to notebook --execute --stdout \
		--ExecutePreprocessor.timeout=1800 \
		notebooks/exploratory/01-data-and-leakage-analysis.ipynb > /dev/null
	$(RUN) jupyter nbconvert --to notebook --execute --stdout \
		--ExecutePreprocessor.timeout=1800 \
		notebooks/exploratory/02-model-evaluation.ipynb > /dev/null

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

.PHONY: docker-build
docker-build: ## Build the container image
	docker build -t term-deposit-propensity:latest .

.PHONY: docker-test
docker-test: ## Run the test suite inside the container
	docker compose run --rm test

.PHONY: docker-train
docker-train: ## Run training inside the container
	docker compose run --rm train

# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #

.PHONY: clean
clean: ## Remove caches and generated outputs (keeps the raw dataset)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf artifacts/*/ artifacts/latest reports/figures/* reports/metrics/*
	@echo "Cleaned. Raw data left in place; use 'make clean-all' to remove it too."

.PHONY: clean-all
clean-all: clean ## Also remove the downloaded dataset
	rm -rf data/raw/* data/interim/* data/processed/*
