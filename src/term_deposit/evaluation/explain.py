"""Feature attribution.

Two levels, with different reliability, kept separate on purpose:

* **Permutation importance** on the held-out split. Model-agnostic and measured
  against the metric that matters. Impurity-based importance — which the original
  notebook used to conclude that ``age`` was the strongest predictor — is biased
  towards high-cardinality continuous features, and the gradient-boosted models in
  the same notebook ranked ``age`` roughly twentieth. Permutation importance does
  not have that bias.
* **SHAP**, optional, for direction and per-customer explanations. Requires the
  ``explain`` extra.

Both are computed on transformed feature names pulled from the fitted pipeline,
so the labels always match the columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from term_deposit.features.pipeline import ENCODE_STEP, output_feature_names
from term_deposit.models.registry import MODEL_STEP
from term_deposit.utils.logging import get_logger
from term_deposit.utils.seeding import make_rng

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = get_logger(__name__)

_MISSING_SHAP = (
    "shap is required for SHAP explanations. Install the explain extra: `uv sync --extra explain`."
)


def permutation_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Permutation importance on the *raw* input columns.

    Permuting raw columns rather than one-hot outputs keeps a categorical
    variable's importance in one place instead of splitting it across its levels,
    which is what a stakeholder actually wants to read.

    Args:
        pipeline: A fitted pipeline.
        X: Held-out features. Never the training split — importance measured on
            training data reflects memorisation.
        y: Held-out labels.
        scoring: Metric whose degradation defines importance.
        n_repeats: Permutations per column.
        random_state: Seed.
        n_jobs: Parallelism.

    Returns:
        Columns ``feature``, ``importance``, ``importance_std``, descending.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def native_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Model-native importance on transformed features, when the estimator has it.

    Reported for comparison only. For tree ensembles this is impurity-based and
    carries the bias described in the module docstring, so it is labelled as such
    wherever it appears.
    """
    estimator = pipeline.named_steps[MODEL_STEP]
    names = output_feature_names(_preprocessor_of(pipeline))

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity_or_gain"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
        kind = "abs_coefficient"
    else:
        msg = f"{type(estimator).__name__} exposes no native feature importance"
        raise AttributeError(msg)

    if len(values) != len(names):
        msg = (
            f"importance vector length ({len(values)}) does not match the number of "
            f"transformed features ({len(names)})"
        )
        raise ValueError(msg)

    return (
        pd.DataFrame({"feature": names, "importance": values, "kind": kind})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _preprocessor_of(pipeline: Pipeline) -> Pipeline:
    """The preprocessing prefix of a full model pipeline."""
    steps = [(name, step) for name, step in pipeline.steps if name != MODEL_STEP]
    if not steps or steps[-1][0] != ENCODE_STEP:
        msg = "pipeline does not have the expected preprocessing layout"
        raise ValueError(msg)
    return Pipeline(steps=steps)


def transform_for_explanation(
    pipeline: Pipeline,
    X: pd.DataFrame,  # noqa: N803
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the fitted preprocessing and return the matrix with its column names."""
    preprocessor = _preprocessor_of(pipeline)
    matrix = np.asarray(preprocessor.transform(X), dtype=float)
    return matrix, output_feature_names(preprocessor)


def shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,  # noqa: N803
    *,
    background_size: int = 200,
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], float]:
    """SHAP values for a tree model inside a fitted pipeline.

    Both the background sample and the explained sample are drawn with an explicit
    seeded generator. The original notebook used the unseeded global RNG for the
    background, which made its published explanation plots unreproducible.

    Args:
        pipeline: A fitted pipeline whose final step is a tree ensemble.
        X: Rows to explain, in raw form.
        background_size: Rows used as the explainer's reference distribution.
        sample_size: Cap on explained rows. ``None`` explains all of them.
        random_state: Seed for both draws.

    Returns:
        ``(shap_matrix, explained_matrix, feature_names, expected_value)``.

    Raises:
        ImportError: If ``shap`` is not installed.
    """
    try:
        import shap
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(_MISSING_SHAP) from error

    matrix, names = transform_for_explanation(pipeline, X)
    rng = make_rng(random_state)

    background_rows = min(background_size, matrix.shape[0])
    background = matrix[rng.choice(matrix.shape[0], background_rows, replace=False)]

    explained = matrix
    if sample_size is not None and matrix.shape[0] > sample_size:
        explained = matrix[rng.choice(matrix.shape[0], sample_size, replace=False)]
        logger.info("Explaining a %d-row sample of %d", sample_size, matrix.shape[0])

    explainer = shap.TreeExplainer(pipeline.named_steps[MODEL_STEP], data=background)
    raw = explainer.shap_values(explained)
    values = np.asarray(raw[1] if isinstance(raw, list) else raw, dtype=float)

    expected = explainer.expected_value
    if isinstance(expected, np.ndarray | list):
        expected = float(np.asarray(expected).ravel()[-1])
    return values, explained, names, float(expected)


def shap_importance(values: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, descending."""
    return (
        pd.DataFrame({"feature": list(names), "importance": np.abs(values).mean(axis=0)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def group_importance(importances: pd.DataFrame, groups: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Aggregate feature importance into named groups.

    Used to state the macro block's total share in one number, which is the
    comparison the project's argument turns on.
    """
    lookup: dict[str, str] = {}
    for group, members in groups.items():
        for member in members:
            lookup[member] = group

    def resolve(feature: str) -> str:
        if feature in lookup:
            return lookup[feature]
        # One-hot outputs are named "<column>_<level>"; match the longest prefix.
        matches = [column for column in lookup if feature.startswith(f"{column}_")]
        return lookup[max(matches, key=len)] if matches else "other"

    grouped = importances.assign(group=importances["feature"].map(resolve))
    total = float(grouped["importance"].sum())
    return (
        grouped.groupby("group", observed=True)["importance"]
        .sum()
        .reset_index()
        .assign(share=lambda d: d["importance"] / total if total else np.nan)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explanation_for_row(
    values: np.ndarray,
    matrix: np.ndarray,
    names: Sequence[str],
    expected_value: float,
    index: int,
) -> Any:
    """Build a ``shap.Explanation`` for one row, ready for a waterfall plot."""
    try:
        import shap
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(_MISSING_SHAP) from error

    return shap.Explanation(
        values=values[index],
        base_values=expected_value,
        data=matrix[index],
        feature_names=list(names),
    )
