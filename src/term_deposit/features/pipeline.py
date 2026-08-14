"""The preprocessing pipeline shared by training, evaluation and inference.

There is one builder, and it returns a *new* estimator on every call. The original
notebook created a single `ColumnTransformer` and passed the same object into four
`Pipeline`s; scikit-learn pipelines hold a reference rather than a copy, so all
four shared one mutable fitted state and a stray `fit_transform` in a later cell
silently refitted the transformer inside every trained model. Returning a fresh
object removes that whole class of bug.

Preprocessing lives inside the estimator, never beside it, so the fit happens on
training folds only and the exact same transformation is replayed at inference.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from term_deposit.config import FeatureConfig
from term_deposit.features.transformers import ColumnSelector, PdaysSentinelEncoder

#: Step names, referenced by the evaluation and explainability code.
SELECT_STEP = "select"
PDAYS_STEP = "pdays"
ENCODE_STEP = "encode"


def resolved_numeric_columns(config: FeatureConfig) -> tuple[str, ...]:
    """Numeric column names *after* the sentinel encoder has run.

    ``pdays`` becomes two columns when :attr:`FeatureConfig.encode_pdays_sentinel`
    is on, so the `ColumnTransformer` must be told about the post-expansion names.
    """
    columns = config.numeric_columns()
    if not config.encode_pdays_sentinel or "pdays" not in columns:
        return columns
    expanded: list[str] = []
    for name in columns:
        if name == "pdays":
            expanded.extend(("pdays_never_contacted", "pdays_days_since_contact"))
        else:
            expanded.append(name)
    return tuple(expanded)


def build_preprocessor(config: FeatureConfig) -> Pipeline:
    """Build a fresh, unfitted preprocessing pipeline.

    The stages are:

    1. :class:`~term_deposit.features.transformers.ColumnSelector` — pin the input
       contract so extra or reordered columns cannot change the result.
    2. :class:`~term_deposit.features.transformers.PdaysSentinelEncoder` — optional;
       turn the ``999`` sentinel into a flag plus a real duration.
    3. ``ColumnTransformer`` — standardise numerics, one-hot encode categoricals.

    Args:
        config: Feature settings.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`. A new instance every call.
    """
    numeric = list(resolved_numeric_columns(config))
    categorical = list(config.categorical_columns())

    encoder = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler() if config.scale_numeric else "passthrough",
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown=config.handle_unknown_categories,
                    sparse_output=False,
                    dtype=np.float64,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps: list[tuple[str, object]] = [(SELECT_STEP, ColumnSelector(config.input_columns()))]
    if config.encode_pdays_sentinel and "pdays" in config.numeric_columns():
        steps.append((PDAYS_STEP, PdaysSentinelEncoder()))
    steps.append((ENCODE_STEP, encoder))

    return Pipeline(steps=steps)


def output_feature_names(preprocessor: Pipeline) -> tuple[str, ...]:
    """Names of the columns a fitted preprocessor produces, in output order.

    Args:
        preprocessor: A pipeline returned by :func:`build_preprocessor`, already fitted.

    Returns:
        One name per output column, aligned with the transformed matrix. These are
        the labels used by feature-importance and SHAP reporting; deriving them
        from the fitted object rather than rebuilding them by hand is what keeps
        the two aligned.

    Raises:
        RuntimeError: If the preprocessor has not been fitted.
    """
    try:
        return tuple(str(name) for name in preprocessor.get_feature_names_out())
    except Exception as error:  # sklearn raises NotFittedError or AttributeError
        msg = "preprocessor is not fitted; call fit before requesting feature names"
        raise RuntimeError(msg) from error
