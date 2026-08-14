"""Dataset vocabulary shared by every stage of the pipeline.

Everything the code knows about the *shape* of the UCI Bank Marketing dataset
lives here, so that a change in the source data surfaces as one edit in one file
rather than as a scatter of string literals across modules.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Raw dataset
# --------------------------------------------------------------------------- #

RAW_CSV_SEPARATOR: Final = ";"
RAW_FILENAME: Final = "bank-additional-full.csv"
RAW_ROW_COUNT: Final = 41_188

#: Public download location. The outer archive nests ``bank-additional.zip``.
UCI_ARCHIVE_URL: Final = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_INNER_ARCHIVE: Final = "bank-additional.zip"
UCI_INNER_MEMBER: Final = "bank-additional/bank-additional-full.csv"

#: SHA-256 of the extracted ``bank-additional-full.csv``. Verified on download so
#: that a silently changed upstream file fails loudly instead of shifting metrics.
RAW_SHA256: Final = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

TARGET_COLUMN: Final = "y"
TARGET_POSITIVE_LABEL: Final = "yes"
TARGET_NEGATIVE_LABEL: Final = "no"

#: Derived binary target produced by the loader.
LABEL_COLUMN: Final = "subscribed"

#: Derived monthly calendar key produced by :mod:`term_deposit.features.calendar`.
PERIOD_COLUMN: Final = "contact_period"

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #

#: Known only *after* a call ends. Using it would leak the outcome into a model
#: whose entire purpose is to decide whom to call. See docs/methodology.md.
POST_OUTCOME_COLUMNS: Final[tuple[str, ...]] = ("duration",)

#: Client attributes available from the CRM before any contact is attempted.
CLIENT_NUMERIC_FEATURES: Final[tuple[str, ...]] = ("age",)

CLIENT_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
)

#: Contact-history attributes of the current and previous campaigns.
CAMPAIGN_NUMERIC_FEATURES: Final[tuple[str, ...]] = ("campaign", "pdays", "previous")

CAMPAIGN_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)

#: Macro indicators. Constant (or near constant) within a calendar month, which
#: makes them a de-facto period identifier — the core finding of this project.
#: See ADR 0003 and ``docs/methodology.md``.
MACRO_FEATURES: Final[tuple[str, ...]] = (
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    *CLIENT_NUMERIC_FEATURES,
    *CAMPAIGN_NUMERIC_FEATURES,
    *MACRO_FEATURES,
)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    *CLIENT_CATEGORICAL_FEATURES,
    *CAMPAIGN_CATEGORICAL_FEATURES,
)

ALL_FEATURES: Final[tuple[str, ...]] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# --------------------------------------------------------------------------- #
# Domain vocabulary
# --------------------------------------------------------------------------- #

#: ``pdays == 999`` is the dataset's sentinel for "never contacted before".
#: Left as a raw number it would be scaled as if 999 days had elapsed.
PDAYS_NEVER_CONTACTED: Final = 999

#: Literal ``"unknown"`` is a real category in this dataset (a refusal or a gap in
#: the CRM), not a missing value. It is preserved rather than imputed. See ADR 0004.
UNKNOWN_CATEGORY: Final = "unknown"

MONTH_ABBREVIATIONS: Final[tuple[str, ...]] = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)  # fmt: skip

MONTH_TO_NUMBER: Final[dict[str, int]] = {
    name: number for number, name in enumerate(MONTH_ABBREVIATIONS, start=1)
}

#: The campaign's first calendar month. The raw file carries no year column; rows
#: are ordered by contact date, so the year is reconstructed from month rollovers.
CAMPAIGN_START_YEAR: Final = 2008
CAMPAIGN_START_MONTH: Final = 5

CATEGORY_DOMAINS: Final[dict[str, tuple[str, ...]]] = {
    "job": (
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
    ),
    "marital": ("divorced", "married", "single", "unknown"),
    "education": (
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ),
    "default": ("no", "yes", "unknown"),
    "housing": ("no", "yes", "unknown"),
    "loan": ("no", "yes", "unknown"),
    "contact": ("cellular", "telephone"),
    "month": MONTH_ABBREVIATIONS,
    "day_of_week": ("mon", "tue", "wed", "thu", "fri"),
    "poutcome": ("failure", "nonexistent", "success"),
}
