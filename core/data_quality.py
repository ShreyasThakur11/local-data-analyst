"""
Data quality checks and non-destructive transformation management.
All transformations are logged and require explicit user approval.
Original DataFrames are never modified.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    issue_type: str
    severity: str          # 'high', 'medium', 'low'
    columns: List[str]
    description: str
    affected_rows: int
    total_rows: int
    recommendation: str
    pct_affected: float = 0.0

    def __post_init__(self) -> None:
        if self.total_rows > 0:
            self.pct_affected = round(self.affected_rows / self.total_rows * 100, 1)


@dataclass
class TransformationRecord:
    operation: str
    columns: List[str]
    parameters: Dict[str, Any]
    rows_affected: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    approved: bool = True


def audit_data_quality(df: pd.DataFrame) -> List[QualityIssue]:
    """
    Run comprehensive data quality checks.
    Returns a list of QualityIssue objects sorted by severity.
    """
    issues: List[QualityIssue] = []
    n = len(df)

    # 1. Missing values per column
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        pct = missing / n * 100
        severity = "high" if pct > 50 else ("medium" if pct > 10 else "low")
        issues.append(
            QualityIssue(
                issue_type="missing_values",
                severity=severity,
                columns=[col],
                description=f"'{col}': {missing:,} missing ({pct:.1f}%)",
                affected_rows=missing,
                total_rows=n,
                recommendation=(
                    "Investigate why values are missing. "
                    "Consider imputation or exclusion based on analytical context."
                ),
            )
        )

    # 2. Duplicate rows
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        issues.append(
            QualityIssue(
                issue_type="duplicate_rows",
                severity="high" if dup_count / n > 0.1 else "medium",
                columns=list(df.columns),
                description=f"{dup_count:,} fully duplicate rows detected.",
                affected_rows=dup_count,
                total_rows=n,
                recommendation="Review duplicate rows — they may represent data entry errors or accidental duplicates.",
            )
        )

    # 3. Constant columns
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            issues.append(
                QualityIssue(
                    issue_type="constant_column",
                    severity="low",
                    columns=[col],
                    description=f"'{col}' has ≤ 1 unique non-null value.",
                    affected_rows=n,
                    total_rows=n,
                    recommendation="Consider excluding this column from analysis.",
                )
            )

    # 4. Mixed-type columns (object columns with numeric data)
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(500)
        if sample.empty:
            continue
        num_parseable = pd.to_numeric(sample.astype(str).str.replace(",", ""), errors="coerce").notna().mean()
        if 0.1 < num_parseable < 0.9:
            issues.append(
                QualityIssue(
                    issue_type="mixed_types",
                    severity="medium",
                    columns=[col],
                    description=f"'{col}' appears to contain mixed numeric and text values.",
                    affected_rows=int((1 - num_parseable) * len(sample)),
                    total_rows=n,
                    recommendation="Review column for data entry inconsistencies.",
                )
            )

    # 5. Invalid dates (string columns that look like dates but fail parsing)
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(200)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            parse_rate = parsed.notna().mean()
            if 0.5 < parse_rate < 1.0:
                invalid = int((1 - parse_rate) * len(sample))
                issues.append(
                    QualityIssue(
                        issue_type="invalid_dates",
                        severity="medium",
                        columns=[col],
                        description=f"'{col}' has ~{invalid} values that failed date parsing.",
                        affected_rows=invalid,
                        total_rows=n,
                        recommendation="Standardise date format or investigate invalid entries.",
                    )
                )
        except Exception:
            pass

    # 6. Potential outliers in numeric columns (IQR × 3 — only extreme ones flagged as quality issues)
    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        extreme = int(((clean < q1 - 3 * iqr) | (clean > q3 + 3 * iqr)).sum())
        if extreme > 0:
            issues.append(
                QualityIssue(
                    issue_type="extreme_outliers",
                    severity="low",
                    columns=[col],
                    description=f"'{col}': {extreme} value(s) beyond 3×IQR fence.",
                    affected_rows=extreme,
                    total_rows=n,
                    recommendation="Investigate extreme values — may be errors or genuine extremes.",
                )
            )

    # Sort: high > medium > low
    sev_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: sev_order.get(x.severity, 3))
    return issues


# ── Safe transformations ───────────────────────────────────────────────────────

class TransformationLog:
    def __init__(self) -> None:
        self.records: List[TransformationRecord] = []

    def add(self, record: TransformationRecord) -> None:
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["timestamp", "operation", "columns", "rows_affected", "description"])
        return pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "operation": r.operation,
                "columns": ", ".join(r.columns),
                "rows_affected": r.rows_affected,
                "description": r.description,
            }
            for r in self.records
        ])


def preview_drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Preview dropping duplicate rows. Returns (result_df, n_dropped)."""
    result = df.drop_duplicates()
    return result, len(df) - len(result)


def preview_drop_rows_with_missing(
    df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 1.0
) -> Tuple[pd.DataFrame, int]:
    """
    Preview removing rows where at least `threshold` fraction of specified columns are missing.
    """
    cols = columns or list(df.columns)
    if threshold >= 1.0:
        mask = df[cols].isna().all(axis=1)
    else:
        mask = df[cols].isna().mean(axis=1) >= threshold

    result = df[~mask]
    return result, int(mask.sum())


def preview_fill_missing(
    df: pd.DataFrame, column: str, method: str = "median"
) -> Tuple[pd.DataFrame, int]:
    """
    Preview filling missing values in a column.
    method: 'mean', 'median', 'mode', 'zero', constant value
    """
    result = df.copy()
    missing_before = int(result[column].isna().sum())

    if method == "mean":
        fill_val = result[column].mean()
    elif method == "median":
        fill_val = result[column].median()
    elif method == "mode":
        modes = result[column].mode()
        fill_val = modes.iloc[0] if not modes.empty else None
    elif method == "zero":
        fill_val = 0
    else:
        # Assume method is a constant value
        try:
            fill_val = float(method)
        except ValueError:
            fill_val = method

    if fill_val is not None:
        result[column] = result[column].fillna(fill_val)

    missing_after = int(result[column].isna().sum())
    return result, missing_before - missing_after


def preview_convert_type(
    df: pd.DataFrame, column: str, target_type: str
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Preview converting a column to a target type.
    Returns (result_df, n_converted, warnings).
    """
    result = df.copy()
    warnings: List[str] = []
    failures = 0

    if target_type == "numeric":
        converted = pd.to_numeric(result[column], errors="coerce")
        failures = int(converted.isna().sum() - result[column].isna().sum())
        result[column] = converted
        if failures:
            warnings.append(f"{failures} values could not be converted to numeric.")

    elif target_type == "datetime":
        converted = pd.to_datetime(result[column], errors="coerce")
        failures = int(converted.isna().sum() - result[column].isna().sum())
        result[column] = converted
        if failures:
            warnings.append(f"{failures} values could not be parsed as dates.")

    elif target_type == "string":
        result[column] = result[column].astype(str)

    elif target_type == "category":
        result[column] = result[column].astype("category")

    return result, len(result) - failures, warnings


def preview_standardise_categories(
    df: pd.DataFrame, column: str, case: str = "title"
) -> pd.DataFrame:
    """Preview standardising category labels (case, strip)."""
    result = df.copy()
    s = result[column].astype(str).str.strip()
    if case == "lower":
        s = s.str.lower()
    elif case == "upper":
        s = s.str.upper()
    elif case == "title":
        s = s.str.title()
    result[column] = s.where(df[column].notna(), other=pd.NA)
    return result
