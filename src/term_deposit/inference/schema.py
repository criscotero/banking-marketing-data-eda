"""The inference input contract.

Training validates a whole table with :mod:`term_deposit.data.schema`; inference
validates one record at a time and has to return a usable error message to a
caller, so it uses pydantic. The two are deliberately different tools for
deliberately different jobs (ADR 0007).

These models are what an HTTP endpoint would validate against, and they are the
reason the scorer cannot silently accept a frame whose ``education`` column has
drifted to free text.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from term_deposit import constants

JobLiteral = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MaritalLiteral = Literal["divorced", "married", "single", "unknown"]
EducationLiteral = Literal[
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
ContactLiteral = Literal["cellular", "telephone"]
MonthLiteral = Literal[
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]
DayLiteral = Literal["mon", "tue", "wed", "thu", "fri"]
PoutcomeLiteral = Literal["failure", "nonexistent", "success"]


class CustomerRecord(BaseModel):
    """One customer as the scorer expects them, *before* any call is placed.

    ``duration`` is deliberately absent. It is the strongest single predictor in
    the raw data and it is only known once the call has ended, so a pre-call
    scorer that accepted it would be scoring the outcome it is meant to predict.
    Sending it is rejected rather than ignored, so the mistake surfaces at the
    boundary instead of quietly inflating an offline metric.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_id: str | None = Field(
        default=None,
        description="Optional caller-supplied identifier, echoed back in the response.",
    )

    age: Annotated[int, Field(ge=17, le=120)]
    job: JobLiteral
    marital: MaritalLiteral
    education: EducationLiteral
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown

    contact: ContactLiteral
    month: MonthLiteral
    day_of_week: DayLiteral
    campaign: Annotated[int, Field(ge=1, description="Contacts in the current campaign.")]
    pdays: Annotated[
        int,
        Field(
            ge=0,
            le=constants.PDAYS_NEVER_CONTACTED,
            description="Days since the last contact; 999 means never contacted.",
        ),
    ]
    previous: Annotated[int, Field(ge=0)]
    poutcome: PoutcomeLiteral

    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: Annotated[float, Field(gt=0, alias="cons.price.idx")]
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: Annotated[float, Field(ge=0)]
    nr_employed: Annotated[float, Field(gt=0, alias="nr.employed")]

    @field_validator("*", mode="before")
    @classmethod
    def _reject_post_call_fields(cls, value: Any) -> Any:
        """Placeholder validator kept for symmetry; ``extra='forbid'`` does the work."""
        return value

    def to_row(self) -> dict[str, Any]:
        """Convert to the raw column names the fitted pipeline expects."""
        payload = self.model_dump(by_alias=True, exclude={"customer_id"})
        return payload


class ScoringRequest(BaseModel):
    """A batch of customers to score."""

    model_config = ConfigDict(extra="forbid")

    records: Annotated[list[CustomerRecord], Field(min_length=1)]

    def to_frame(self) -> pd.DataFrame:
        """Materialise the batch as a DataFrame with raw column names."""
        return pd.DataFrame([record.to_row() for record in self.records])

    def customer_ids(self) -> list[str | None]:
        """Caller-supplied identifiers, positionally aligned with ``records``."""
        return [record.customer_id for record in self.records]


class ScoredCustomer(BaseModel):
    """One scored customer."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = None
    subscription_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    predicted_class: Literal[0, 1]
    tier: str | None = None


class ScoringResponse(BaseModel):
    """A scored batch, stamped with the artifact that produced it.

    The model identity and threshold travel with the scores so that a downstream
    consumer can tell which model produced a ranking months later.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_created_at: str
    decision_threshold: float
    predictions: list[ScoredCustomer]

    def to_frame(self) -> pd.DataFrame:
        """Predictions as a DataFrame."""
        return pd.DataFrame([prediction.model_dump() for prediction in self.predictions])


def validate_frame(frame: pd.DataFrame, *, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Validate a scoring frame row by row against :class:`CustomerRecord`.

    Args:
        frame: Raw input columns.
        required_columns: Columns the fitted pipeline needs, from the artifact's
            metadata.

    Returns:
        The frame restricted to ``required_columns``, in the model's order.

    Raises:
        ValueError: If required columns are missing or any row fails validation.
            Up to five row-level errors are reported at once so that a malformed
            file can be fixed in one pass.
    """
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        msg = f"scoring input is missing required column(s): {missing}"
        raise ValueError(msg)

    errors: list[str] = []
    for position, record in enumerate(frame.to_dict(orient="records")):
        payload = {
            key: value
            for key, value in record.items()
            if key in CustomerRecord.model_fields
            or key in {field.alias for field in CustomerRecord.model_fields.values() if field.alias}
        }
        try:
            CustomerRecord.model_validate(payload)
        except Exception as error:  # pydantic ValidationError
            errors.append(f"row {position}: {error}")
            if len(errors) >= 5:
                break

    if errors:
        joined = "\n  - ".join(errors)
        msg = f"scoring input failed validation:\n  - {joined}"
        raise ValueError(msg)

    return frame.loc[:, list(required_columns)]
