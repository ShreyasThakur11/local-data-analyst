"""
Type inference and semantic role detection.
Determines the most appropriate type and business meaning for each column.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class InferredType(str, Enum):
    NUMERIC_CONTINUOUS = "Numeric (continuous)"
    NUMERIC_DISCRETE = "Numeric (discrete)"
    CATEGORICAL = "Categorical"
    DATE = "Date / Timestamp"
    BOOLEAN = "Boolean"
    TEXT = "Free text"
    IDENTIFIER = "Identifier / Key"
    CURRENCY = "Currency"
    PERCENTAGE = "Percentage"
    GEOGRAPHIC = "Geographic"
    ORDINAL = "Ordinal"
    UNKNOWN = "Unknown / Ambiguous"


class SemanticRole(str, Enum):
    IDENTIFIER = "Identifier"
    DATE = "Date / Timestamp"
    CATEGORY = "Category / Dimension"
    MEASURE = "Numeric Measure"
    OUTCOME = "Outcome / Target"
    GROUP = "Grouping Variable"
    WEIGHT = "Weight"
    GEOGRAPHY = "Geography"
    TEXT = "Free Text"
    CURRENCY = "Currency"
    PERCENTAGE = "Percentage"
    DURATION = "Duration"
    QUANTITY = "Quantity"
    BOOLEAN = "Boolean"
    EXCLUDE = "Exclude from Analysis"
    UNKNOWN = "Unknown"


@dataclass
class ColumnProfile:
    name: str
    original_type: str
    inferred_type: InferredType
    semantic_role: SemanticRole
    non_missing: int
    missing_count: int
    missing_pct: float
    unique_count: int
    duplicate_count: int
    example_values: List[Any]
    notes: List[str] = field(default_factory=list)
    suggested_analyses: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0–1; lower = more ambiguous
    is_override: bool = False  # True when user has manually changed classification


# ── Identifier name hints ──────────────────────────────────────────────────────

_ID_PATTERNS = re.compile(
    r"(\bid\b|_id$|^id_|code$|^code_|_code$|_key$|^key_|sku|isbn|ean|barcode|"
    r"zip|postal|postcode|pin_code|account_no|account_number|employee_no|"
    r"order_no|invoice_no|reference|ref_no|batch_no|serial|license_plate)",
    re.IGNORECASE,
)

_DATE_PATTERNS = re.compile(
    r"(date|time|year|month|quarter|week|day|timestamp|period|fiscal|"
    r"created_at|updated_at|modified|reported)",
    re.IGNORECASE,
)

_CURRENCY_PATTERNS = re.compile(
    r"(price|cost|revenue|sales|profit|loss|amount|fee|charge|"
    r"budget|spend|expenditure|income|salary|wage|turnover|margin|"
    r"capex|opex|value|rate|tax)",
    re.IGNORECASE,
)

_PERCENTAGE_PATTERNS = re.compile(
    r"(pct|percent|ratio|rate|share|proportion|utilisation|utilization|"
    r"efficiency|fill_rate|on_time|availability|yield)",
    re.IGNORECASE,
)

_GEO_PATTERNS = re.compile(
    r"(country|region|state|province|city|district|zone|location|"
    r"latitude|longitude|lat|lon|lng|address|territory|plant_loc)",
    re.IGNORECASE,
)

_BOOL_PATTERNS = re.compile(
    r"(is_|has_|flag|active|enabled|approved|valid|deleted|confirmed|shipped)",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"^year$|_year$|^yr$|_yr$", re.IGNORECASE)


def infer_column(series: pd.Series, name: str) -> ColumnProfile:
    """
    Infer type, semantic role, and basic statistics for a single column.
    """
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_non_missing = n_total - n_missing
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    n_duplicate = n_non_missing - n_unique

    examples = _safe_examples(non_null)
    original_type = str(series.dtype)
    notes: List[str] = []
    suggested: List[str] = []
    confidence = 1.0

    # ── Type inference ────────────────────────────────────────────────────────

    inferred = _infer_type(series, non_null, name, notes)

    # ── Semantic role ────────────────────────────────────────────────────────

    role = _infer_role(series, non_null, name, inferred, n_unique, n_non_missing, notes)

    # ── Suggested analyses ────────────────────────────────────────────────────

    suggested = _suggest_analyses(inferred, role, n_non_missing, n_unique)

    missing_pct = round(n_missing / n_total * 100, 1) if n_total else 0.0

    return ColumnProfile(
        name=name,
        original_type=original_type,
        inferred_type=inferred,
        semantic_role=role,
        non_missing=n_non_missing,
        missing_count=n_missing,
        missing_pct=missing_pct,
        unique_count=n_unique,
        duplicate_count=n_duplicate,
        example_values=examples,
        notes=notes,
        suggested_analyses=suggested,
        confidence=confidence,
    )


def infer_all_columns(df: pd.DataFrame) -> Dict[str, ColumnProfile]:
    """Infer types and roles for every column in a DataFrame."""
    return {col: infer_column(df[col], col) for col in df.columns}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _safe_examples(series: pd.Series, n: int = 5) -> List[Any]:
    try:
        return [v for v in series.dropna().head(n).tolist()]
    except Exception:
        return []


def _infer_type(
    series: pd.Series, non_null: pd.Series, name: str, notes: List[str]
) -> InferredType:
    dtype = series.dtype

    # Already a datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return InferredType.DATE

    # Boolean
    if pd.api.types.is_bool_dtype(dtype):
        return InferredType.BOOLEAN

    # Numeric pandas type
    if pd.api.types.is_numeric_dtype(dtype):
        return _numeric_subtype(series, non_null, name, notes)

    # Object/string — try conversions
    if dtype == object or pd.api.types.is_string_dtype(dtype):
        return _infer_from_string(series, non_null, name, notes)

    return InferredType.UNKNOWN


def _numeric_subtype(
    series: pd.Series, non_null: pd.Series, name: str, notes: List[str]
) -> InferredType:
    # Year columns should not be treated as continuous measurements
    if _YEAR_PATTERN.search(name):
        notes.append(f"'{name}' looks like a year — treated as discrete, not continuous.")
        return InferredType.NUMERIC_DISCRETE

    # Identifier patterns even if numeric
    if _ID_PATTERNS.search(name):
        notes.append(f"'{name}' matches identifier naming pattern.")
        return InferredType.IDENTIFIER

    # All integer values?
    if non_null.dropna().apply(lambda x: float(x) == int(x) if pd.notna(x) else True).all():
        n_unique = non_null.nunique()
        n_total = len(non_null)
        uniqueness = n_unique / n_total if n_total else 0

        # High uniqueness → likely identifier
        if uniqueness > 0.9 and n_unique > 50:
            notes.append(f"High uniqueness ratio ({uniqueness:.1%}) → may be an identifier.")
            return InferredType.IDENTIFIER

        return InferredType.NUMERIC_DISCRETE

    return InferredType.NUMERIC_CONTINUOUS


def _infer_from_string(
    series: pd.Series, non_null: pd.Series, name: str, notes: List[str]
) -> InferredType:
    if non_null.empty:
        return InferredType.UNKNOWN

    # Boolean-like?
    unique_lower = set(non_null.astype(str).str.lower().unique())
    bool_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"1", "0"},
        {"y", "n"},
        {"t", "f"},
    ]
    for bs in bool_sets:
        if unique_lower <= bs:
            return InferredType.BOOLEAN

    # Try date parsing
    sample = non_null.head(200)
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.8:
            notes.append(f"'{name}' appears to contain date strings — conversion recommended.")
            return InferredType.DATE
    except Exception:
        pass

    # Try numeric
    try:
        num_sample = non_null.head(200).astype(str).str.replace(",", "").str.replace("$", "").str.replace("%", "")
        converted = pd.to_numeric(num_sample, errors="coerce")
        if converted.notna().mean() > 0.85:
            notes.append(f"'{name}' is stored as text but appears numeric — conversion recommended.")
            return InferredType.NUMERIC_CONTINUOUS
    except Exception:
        pass

    # Cardinality
    n_unique = non_null.nunique()
    n_total = len(non_null)
    uniqueness = n_unique / n_total if n_total else 0

    if uniqueness > 0.95:
        avg_len = non_null.astype(str).str.len().mean()
        if avg_len > 30:
            return InferredType.TEXT
        return InferredType.IDENTIFIER

    if n_unique <= max(30, n_total * 0.05):
        return InferredType.CATEGORICAL

    return InferredType.TEXT


def _infer_role(
    series: pd.Series,
    non_null: pd.Series,
    name: str,
    inferred: InferredType,
    n_unique: int,
    n_non_missing: int,
    notes: List[str],
) -> SemanticRole:

    # Date
    if inferred == InferredType.DATE:
        return SemanticRole.DATE

    # Boolean
    if inferred == InferredType.BOOLEAN:
        return SemanticRole.BOOLEAN

    # Identifier
    if inferred == InferredType.IDENTIFIER or _ID_PATTERNS.search(name):
        return SemanticRole.IDENTIFIER

    # Geographic
    if _GEO_PATTERNS.search(name):
        return SemanticRole.GEOGRAPHY

    # Date name hints (if type didn't catch it)
    if _DATE_PATTERNS.search(name) and inferred not in (InferredType.NUMERIC_CONTINUOUS,):
        return SemanticRole.DATE

    # Currency
    if _CURRENCY_PATTERNS.search(name) and inferred in (
        InferredType.NUMERIC_CONTINUOUS,
        InferredType.NUMERIC_DISCRETE,
    ):
        return SemanticRole.CURRENCY

    # Percentage
    if _PERCENTAGE_PATTERNS.search(name) and inferred in (
        InferredType.NUMERIC_CONTINUOUS,
        InferredType.NUMERIC_DISCRETE,
    ):
        # Also detect values 0–1 or 0–100
        if not non_null.empty and pd.api.types.is_numeric_dtype(series):
            mx = non_null.max()
            if mx <= 1.01:
                notes.append(f"'{name}' values are 0–1; may represent proportions.")
            elif mx <= 100.01:
                notes.append(f"'{name}' values are 0–100; may represent percentages.")
        return SemanticRole.PERCENTAGE

    # Boolean name hints
    if _BOOL_PATTERNS.search(name):
        return SemanticRole.BOOLEAN

    # Numeric measures
    if inferred in (InferredType.NUMERIC_CONTINUOUS, InferredType.CURRENCY):
        return SemanticRole.MEASURE

    if inferred == InferredType.NUMERIC_DISCRETE:
        # Could be quantity or category
        if n_unique <= 20:
            return SemanticRole.CATEGORY
        return SemanticRole.QUANTITY

    if inferred == InferredType.CATEGORICAL:
        return SemanticRole.CATEGORY

    if inferred == InferredType.TEXT:
        return SemanticRole.TEXT

    return SemanticRole.UNKNOWN


def _suggest_analyses(
    inferred: InferredType,
    role: SemanticRole,
    n_non_missing: int,
    n_unique: int,
) -> List[str]:
    suggestions: List[str] = []

    if inferred in (InferredType.NUMERIC_CONTINUOUS, InferredType.CURRENCY):
        suggestions += [
            "Histogram and distribution plot",
            "Box plot for outlier detection",
            "Scatter plot vs. another numeric column",
            "Correlation analysis",
            "Descriptive statistics (mean, median, IQR)",
        ]
    elif inferred == InferredType.NUMERIC_DISCRETE:
        suggestions += [
            "Frequency distribution",
            "Bar chart of top values",
        ]
    elif inferred == InferredType.CATEGORICAL:
        suggestions += [
            "Frequency count and bar chart",
            "Cross-tabulation vs. another category",
            "Group comparison of numeric measures",
        ]
    elif inferred == InferredType.DATE:
        suggestions += [
            "Time-series trend plot",
            "Period-over-period change",
            "Record count by month/quarter",
        ]
    elif inferred == InferredType.BOOLEAN:
        suggestions += [
            "Proportion analysis",
            "Group comparison",
        ]
    elif inferred == InferredType.TEXT:
        suggestions += [
            "Text length distribution",
            "Blank/empty value analysis",
            "Common pattern detection",
        ]
    elif inferred == InferredType.IDENTIFIER:
        suggestions += [
            "Uniqueness check",
            "Duplicate key detection",
        ]

    if n_non_missing < 30:
        suggestions.append("⚠️ Very small sample — interpret results cautiously.")

    return suggestions
