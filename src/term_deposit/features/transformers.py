"""Custom scikit-learn transformers.

Each one is a stateless-by-design estimator so that it can live inside a
`Pipeline`, be cloned by cross-validation, and be pickled with the trained model.
That last point is what keeps training and inference from drifting apart: there
is exactly one implementation of every transformation and it travels with the
artifact.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from term_deposit import constants

# scikit-learn ships no type information, so mypy sees its base classes as `Any`
# and flags every subclass. Subclassing them is the documented way to build a
# transformer that `Pipeline` and `clone` understand.
# mypy: disable-error-code="misc"


class PdaysSentinelEncoder(BaseEstimator, TransformerMixin):
    """Split ``pdays`` into a contact flag and a real elapsed-days column.

    ``pdays == 999`` is the dataset's code for "this client was never contacted in
    a previous campaign". Left alone it is scaled as though 999 days had elapsed,
    which places 96% of clients at one extreme of a variable whose ordering is
    meaningless for them, and lets a linear model read the sentinel as a distance.

    The encoder emits two columns instead:

    * ``pdays_never_contacted`` — 1 when the sentinel is present.
    * ``pdays_days_since_contact`` — the real value, and ``fill_value`` otherwise.

    The transformation is a fixed rule, not something learned from data, so it
    cannot leak: `fit` only records the input column layout.

    Args:
        column: Name of the sentinel-bearing column.
        sentinel: The sentinel value.
        fill_value: Value substituted for the sentinel in the numeric column. The
            default 0 pairs with the flag: "0 days, and the flag says it never happened".

    Example:
        >>> frame = pd.DataFrame({"pdays": [999, 3], "age": [40, 51]})
        >>> PdaysSentinelEncoder().fit_transform(frame)[
        ...     ["pdays_never_contacted", "pdays_days_since_contact"]
        ... ].to_dict("list")
        {'pdays_never_contacted': [1, 0], 'pdays_days_since_contact': [0.0, 3.0]}
    """

    def __init__(
        self,
        column: str = "pdays",
        sentinel: int = constants.PDAYS_NEVER_CONTACTED,
        fill_value: float = 0.0,
    ) -> None:
        """Store the encoding rule. See the class docstring for the arguments."""
        self.column = column
        self.sentinel = sentinel
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:  # noqa: ARG002, N803
        """Record the input columns. No statistics are learned from the data."""
        if self.column not in X.columns:
            msg = f"PdaysSentinelEncoder expects a {self.column!r} column, got {list(X.columns)}"
            raise ValueError(msg)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Replace the sentinel column with a flag plus a numeric column."""
        check_is_fitted(self, "feature_names_in_")
        result = X.copy()
        values = pd.to_numeric(result[self.column], errors="coerce")
        is_sentinel = values.eq(self.sentinel)

        # `get_loc` can return a slice or a mask for duplicated labels. The raw
        # dataset has unique column names, so anything else means the caller
        # handed us a malformed frame and should hear about it.
        position = result.columns.get_loc(self.column)
        if not isinstance(position, int):
            msg = f"column {self.column!r} appears more than once in the input frame"
            raise ValueError(msg)
        result = result.drop(columns=[self.column])
        result.insert(position, f"{self.column}_never_contacted", is_sentinel.astype("int8"))
        result.insert(
            position + 1,
            f"{self.column}_days_since_contact",
            values.where(~is_sentinel, self.fill_value).astype("float64"),
        )
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Output column names, with the sentinel column expanded in place."""
        check_is_fitted(self, "feature_names_in_")
        names = list(input_features) if input_features is not None else list(self.feature_names_in_)
        position = names.index(self.column)
        return np.asarray(
            [
                *names[:position],
                f"{self.column}_never_contacted",
                f"{self.column}_days_since_contact",
                *names[position + 1 :],
            ],
            dtype=object,
        )

    def _more_tags(self) -> dict[str, bool]:
        return {"stateless": True, "no_validation": True}


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select and reorder columns by name.

    Placed at the head of the pipeline so that the artifact carries its own input
    contract: a scoring frame with extra columns, or with columns in a different
    order, produces the same result as the training frame did.

    Args:
        columns: Columns to keep, in the order the model expects them.
    """

    def __init__(self, columns: tuple[str, ...] | list[str]) -> None:
        """Store the column list to select, in the order the model expects."""
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:  # noqa: ARG002, N803
        """Validate that every required column is present."""
        missing = [name for name in self.columns if name not in X.columns]
        if missing:
            msg = f"missing required column(s): {missing}"
            raise ValueError(msg)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return the configured columns in the configured order."""
        check_is_fitted(self, "feature_names_in_")
        missing = [name for name in self.columns if name not in X.columns]
        if missing:
            msg = f"missing required column(s) at transform time: {missing}"
            raise ValueError(msg)
        return X.loc[:, list(self.columns)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:  # noqa: ARG002
        """The selected column names."""
        return np.asarray(self.columns, dtype=object)
