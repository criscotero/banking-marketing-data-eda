"""The scoring surface.

`Predictor` is the only supported way to use a trained model. It loads the
artifact once, validates input against the contract recorded in that artifact's
metadata, and applies the threshold the model was shipped with.

There is no second copy of the preprocessing here. The fitted pipeline carries
its own transformations, which is what guarantees that a score produced in
production matches the one produced during evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from term_deposit.config import InferenceConfig
from term_deposit.evaluation.thresholds import assign_tiers
from term_deposit.inference.schema import (
    ScoredCustomer,
    ScoringRequest,
    ScoringResponse,
    validate_frame,
)
from term_deposit.models.persistence import (
    ArtifactError,
    ModelArtifact,
    load_artifact,
    resolve_artifact_path,
)
from term_deposit.utils.logging import get_logger

logger = get_logger(__name__)


class PredictorError(RuntimeError):
    """Raised when scoring cannot proceed."""


class Predictor:
    """Score customers with a persisted model.

    Args:
        artifact: A loaded model artifact.
        config: Inference settings (batching, tiering, validation).

    Example:
        >>> predictor = load_predictor(Path("artifacts"), InferenceConfig())  # doctest: +SKIP
        >>> predictor.score_frame(frame).head()  # doctest: +SKIP
    """

    def __init__(self, artifact: ModelArtifact, config: InferenceConfig | None = None) -> None:
        """Bind a loaded artifact to its inference settings.

        Raises:
            PredictorError: If the artifact does not record its input contract.
        """
        self.artifact = artifact
        self.config = config or InferenceConfig()
        self._input_columns = tuple(artifact.metadata.input_columns)
        if not self._input_columns:
            msg = (
                "artifact metadata does not record its input columns; "
                "retrain with the current pipeline to produce a usable artifact"
            )
            raise PredictorError(msg)

    @property
    def threshold(self) -> float:
        """The decision threshold the model was shipped with."""
        return float(self.artifact.metadata.decision_threshold)

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Raw columns the model requires, in order."""
        return self._input_columns

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities for a raw feature frame.

        Args:
            frame: Raw columns. Validated when ``config.validate_input`` is set.

        Returns:
            One probability per row.

        Raises:
            ValueError: If validation is enabled and the input violates the contract.
        """
        prepared = (
            validate_frame(frame, required_columns=self._input_columns)
            if self.config.validate_input
            else frame.loc[:, list(self._input_columns)]
        )
        return np.asarray(self.artifact.pipeline.predict_proba(prepared))[:, 1]

    def score_frame(self, frame: pd.DataFrame, *, id_column: str | None = None) -> pd.DataFrame:
        """Score a batch and return probabilities, classes and priority tiers.

        Batching keeps peak memory bounded for large lists; the results are
        identical to scoring the frame in one go because the pipeline is stateless
        at predict time.

        Args:
            frame: Raw input columns, plus an optional identifier column.
            id_column: Column to carry through as ``customer_id``.

        Returns:
            One row per input row, in input order, with ``subscription_probability``,
            ``predicted_class`` and (when configured) ``tier``.
        """
        if frame.empty:
            msg = "cannot score an empty frame"
            raise PredictorError(msg)

        probabilities = np.concatenate(
            [self.predict_proba(chunk) for chunk in self._batches(frame)]
        )
        result = pd.DataFrame(index=frame.index)
        if id_column is not None:
            if id_column not in frame.columns:
                msg = f"id_column {id_column!r} not found in the input frame"
                raise PredictorError(msg)
            result["customer_id"] = frame[id_column].to_numpy()
        result["subscription_probability"] = probabilities
        result["predicted_class"] = (probabilities >= self.threshold).astype(int)

        if self.config.include_tier:
            result["tier"] = assign_tiers(
                probabilities,
                quantiles=self.config.tier_quantiles,
                labels=self.config.tier_labels,
            )

        logger.info(
            "Scored %d record(s); %d above the %.3f threshold",
            len(result),
            int(result["predicted_class"].sum()),
            self.threshold,
        )
        return result

    def score_request(self, request: ScoringRequest) -> ScoringResponse:
        """Score a validated batch and return a typed response.

        This is the shape an HTTP endpoint would expose: a contract in, a
        contract out, with the model's identity attached to the result.
        """
        frame = request.to_frame()
        scored = self.score_frame(frame)
        tiers = scored["tier"] if "tier" in scored.columns else pd.Series([None] * len(scored))
        return ScoringResponse(
            model_name=self.artifact.metadata.model_name,
            model_created_at=self.artifact.metadata.created_at,
            decision_threshold=self.threshold,
            predictions=[
                ScoredCustomer(
                    customer_id=customer_id,
                    subscription_probability=float(probability),
                    predicted_class=int(predicted),
                    tier=tier,
                )
                for customer_id, probability, predicted, tier in zip(
                    request.customer_ids(),
                    scored["subscription_probability"],
                    scored["predicted_class"],
                    tiers,
                    strict=True,
                )
            ],
        )

    def rank(
        self, frame: pd.DataFrame, *, top_k: int | None = None, id_column: str | None = None
    ) -> pd.DataFrame:
        """Return the call list, highest propensity first.

        The deliverable the campaign actually consumes: a ranking, optionally
        truncated to the number of calls the centre can make.
        """
        scored = self.score_frame(frame, id_column=id_column)
        ordered = scored.sort_values("subscription_probability", ascending=False, kind="stable")
        ordered.insert(0, "rank", np.arange(1, len(ordered) + 1))
        return ordered.head(top_k) if top_k else ordered

    def _batches(self, frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Yield row chunks of at most ``config.batch_size``."""
        size = self.config.batch_size
        for start in range(0, len(frame), size):
            yield frame.iloc[start : start + size]


def load_predictor(
    artifacts_dir: Path,
    config: InferenceConfig | None = None,
    *,
    model_id: str | None = None,
) -> Predictor:
    """Load a predictor from the artifacts directory.

    Args:
        artifacts_dir: Directory containing run directories.
        config: Inference settings; defaults are used when omitted.
        model_id: Run directory name, or ``"latest"``. Falls back to
            ``config.model_id``.

    Returns:
        A ready-to-use :class:`Predictor`.

    Raises:
        PredictorError: If no matching artifact exists.
    """
    resolved_config = config or InferenceConfig()
    target = model_id or resolved_config.model_id
    try:
        directory = resolve_artifact_path(artifacts_dir, target)
        artifact = load_artifact(directory)
    except ArtifactError as error:
        raise PredictorError(str(error)) from error
    logger.info("Loaded model %r from %s", artifact.metadata.model_name, directory)
    return Predictor(artifact, resolved_config)
