"""Build estimators from configuration.

A model is a name plus a parameter mapping in YAML; this module turns that into a
scikit-learn compatible object wrapped in the shared preprocessing pipeline.
Keeping construction in one place means a new candidate is a config entry, not a
code change, and that every model in a comparison is guaranteed to have received
the same preprocessing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from term_deposit.config import FeatureConfig, ModelSpec
from term_deposit.features.pipeline import build_preprocessor
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)

#: Pipeline step holding the estimator, referenced by evaluation and explainability.
MODEL_STEP = "model"


class UnknownEstimatorError(ValueError):
    """Raised when a config names an estimator that is not registered."""


def _build_dummy(params: Mapping[str, Any]) -> BaseEstimator:
    """Prior-probability baseline. Any model that cannot beat it is not a model."""
    return DummyClassifier(**{"strategy": "prior", **params})


def _build_logistic_regression(params: Mapping[str, Any]) -> BaseEstimator:
    defaults: dict[str, Any] = {"max_iter": 2000, "solver": "lbfgs"}
    return LogisticRegression(**{**defaults, **params})


def _build_random_forest(params: Mapping[str, Any]) -> BaseEstimator:
    defaults: dict[str, Any] = {"n_estimators": 300}
    return RandomForestClassifier(**{**defaults, **params})


def _build_gradient_boosting(params: Mapping[str, Any]) -> BaseEstimator:
    defaults: dict[str, Any] = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}
    return GradientBoostingClassifier(**{**defaults, **params})


def _build_xgboost(params: Mapping[str, Any]) -> BaseEstimator:
    # Imported lazily so that the rest of the package stays importable if the
    # optional native wheel is unavailable on an exotic platform.
    from xgboost import XGBClassifier

    defaults: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    return XGBClassifier(**{**defaults, **params})


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], BaseEstimator]] = {
    "dummy": _build_dummy,
    "logistic_regression": _build_logistic_regression,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "xgboost": _build_xgboost,
}

#: Estimators that accept ``random_state``. ``DummyClassifier(strategy="prior")``
#: is deterministic and warns if handed one.
_SEEDED_ESTIMATORS = frozenset({"logistic_regression", "random_forest", "gradient_boosting"})

#: Estimators whose seed argument is ``random_state`` under a different meaning.
_SEEDED_XGBOOST = frozenset({"xgboost"})


def build_estimator(
    spec: ModelSpec,
    *,
    random_state: int,
    scale_pos_weight: float | None = None,
    n_jobs: int | None = None,
) -> BaseEstimator:
    """Instantiate the estimator described by ``spec``.

    Imbalance handling is resolved here rather than in YAML because
    ``scale_pos_weight`` depends on the training fold's class balance, which is
    not known until the split exists.

    Args:
        spec: Model definition from configuration.
        random_state: Seed applied to every estimator that accepts one.
        scale_pos_weight: Negative-to-positive ratio of the training split.
            Required when ``spec.balance_strategy == "scale_pos_weight"``.
        n_jobs: Parallelism for estimators that support it.

    Returns:
        An unfitted estimator.

    Raises:
        UnknownEstimatorError: If ``spec.estimator`` is not registered.
        ValueError: If a balance strategy is requested without the data it needs.
    """
    builder = _BUILDERS.get(spec.estimator)
    if builder is None:
        msg = f"unknown estimator {spec.estimator!r}; registered estimators are {sorted(_BUILDERS)}"
        raise UnknownEstimatorError(msg)

    params: dict[str, Any] = dict(spec.params)

    if spec.estimator in _SEEDED_ESTIMATORS | _SEEDED_XGBOOST:
        params.setdefault("random_state", random_state)

    # LogisticRegression dropped n_jobs in scikit-learn 1.8; passing it now warns.
    if n_jobs is not None and spec.estimator in {"random_forest", "xgboost"}:
        params.setdefault("n_jobs", n_jobs)

    if spec.balance_strategy == "class_weight":
        params.setdefault("class_weight", "balanced")
    elif spec.balance_strategy == "scale_pos_weight":
        if scale_pos_weight is None:
            msg = (
                f"model {spec.name!r} requests scale_pos_weight balancing but no "
                "training-split ratio was supplied"
            )
            raise ValueError(msg)
        params.setdefault("scale_pos_weight", scale_pos_weight)

    estimator = builder(params)
    logger.debug("Built %s (%s) with params %s", spec.name, spec.estimator, params)
    return estimator


def build_pipeline(
    spec: ModelSpec,
    features: FeatureConfig,
    *,
    random_state: int,
    scale_pos_weight: float | None = None,
    n_jobs: int | None = None,
) -> Pipeline:
    """Wrap the estimator in a freshly built preprocessing pipeline.

    Returns:
        An unfitted ``Pipeline`` with its own preprocessor instance, so that
        cloning it for cross-validation cannot disturb any other model.
    """
    preprocessor = build_preprocessor(features)
    estimator = build_estimator(
        spec, random_state=random_state, scale_pos_weight=scale_pos_weight, n_jobs=n_jobs
    )
    return Pipeline(steps=[*preprocessor.steps, (MODEL_STEP, estimator)])


def registered_estimators() -> tuple[str, ...]:
    """Names accepted by the ``estimator`` field of a model spec."""
    return tuple(sorted(_BUILDERS))
