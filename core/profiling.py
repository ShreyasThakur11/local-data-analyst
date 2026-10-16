"""
Dataset profiling engine.
Creates comprehensive profiles including data quality flags and recommendations.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.type_inference import ColumnProfile, InferredType, SemanticRole, infer_all_columns

logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    dataset_id: str
    source_name: str
    n_rows: int
    n_cols: int
    columns: Dict[str, ColumnProfile]
    total_missing: int
    total_missing_pct: float
    duplicate_rows: int
    memory_mb: float
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    date_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    bool_cols: List[str] = field(default_factory=list)
    quality_score: float = 100.0   # 0–100
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def shape_str(self) -> str:
        return f"{self.n_rows:,} rows × {self.n_cols} cols"


def profile_dataframe(
    df: pd.DataFrame,
    source_name: str = "dataset",
    sample_limit: int = 50_000,
) -> DatasetProfile:
    """
    Build a comprehensive DatasetProfile from a DataFrame.
    Uses sampling for expensive operations when df exceeds sample_limit rows.
    """
    n_rows, n_cols = df.shape
    is_sampled = n_rows > sample_limit
    sample = df.sample(sample_limit, random_state=42) if is_sampled else df

    # Dataset identity fingerprint
    dataset_id = _fingerprint(df)

    # Per-column inference
    column_profiles = infer_all_columns(sample)

    # Counts
    total_missing = int(df.isna().sum().sum())
    total_missing_pct = round(total_missing / (n_rows * n_cols) * 100, 2) if (n_rows * n_cols) else 0.0

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Memory
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

    # Group columns by type
    numeric_cols, categorical_cols, date_cols, text_cols, id_cols, bool_cols = _group_columns(column_profiles)

    # Quality score
    quality_score, warnings = _compute_quality_score(df, column_profiles, duplicate_rows, total_missing_pct)

    if is_sampled:
        warnings.append(
            f"Profiling used a random sample of {sample_limit:,} / {n_rows:,} rows "
            f"for type inference. Exact statistics may differ slightly."
        )

    # Recommendations
    recommendations = _generate_recommendations(
        df, column_profiles, numeric_cols, categorical_cols, date_cols, n_rows
    )

    return DatasetProfile(
        dataset_id=dataset_id,
        source_name=source_name,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=column_profiles,
        total_missing=total_missing,
        total_missing_pct=total_missing_pct,
        duplicate_rows=duplicate_rows,
        memory_mb=memory_mb,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        date_cols=date_cols,
        text_cols=text_cols,
        id_cols=id_cols,
        bool_cols=bool_cols,
        quality_score=quality_score,
        recommendations=recommendations,
        warnings=warnings,
    )


def _fingerprint(df: pd.DataFrame) -> str:
    token = f"{list(df.columns)}|{df.shape}|{str(df.dtypes.to_dict())}"
    return hashlib.md5(token.encode()).hexdigest()[:12]  # noqa: S324


def _group_columns(profiles: Dict[str, ColumnProfile]):
    numeric_cols, categorical_cols, date_cols, text_cols, id_cols, bool_cols = [], [], [], [], [], []

    for name, cp in profiles.items():
        t = cp.inferred_type
        if t in (InferredType.NUMERIC_CONTINUOUS, InferredType.NUMERIC_DISCRETE,
                 InferredType.CURRENCY, InferredType.PERCENTAGE):
            numeric_cols.append(name)
        elif t == InferredType.CATEGORICAL:
            categorical_cols.append(name)
        elif t == InferredType.DATE:
            date_cols.append(name)
        elif t == InferredType.TEXT:
            text_cols.append(name)
        elif t == InferredType.IDENTIFIER:
            id_cols.append(name)
        elif t == InferredType.BOOLEAN:
            bool_cols.append(name)

    return numeric_cols, categorical_cols, date_cols, text_cols, id_cols, bool_cols


def _compute_quality_score(
    df: pd.DataFrame,
    profiles: Dict[str, ColumnProfile],
    duplicate_rows: int,
    total_missing_pct: float,
) -> Tuple[float, List[str]]:
    score = 100.0
    warnings: List[str] = []
    n_rows = len(df)

    # Missing data penalty
    if total_missing_pct > 30:
        penalty = min(30, total_missing_pct * 0.5)
        score -= penalty
        warnings.append(f"High overall missingness: {total_missing_pct:.1f}% of values are missing.")
    elif total_missing_pct > 10:
        score -= total_missing_pct * 0.3

    # Duplicate rows penalty
    if n_rows > 0:
        dup_pct = duplicate_rows / n_rows * 100
        if dup_pct > 10:
            score -= min(20, dup_pct * 0.5)
            warnings.append(f"{duplicate_rows:,} duplicate rows detected ({dup_pct:.1f}%).")
        elif dup_pct > 1:
            score -= dup_pct * 0.3

    # Constant column penalty
    const_cols = [
        name for name, cp in profiles.items()
        if cp.unique_count <= 1 and cp.non_missing > 0
    ]
    if const_cols:
        score -= len(const_cols) * 2
        warnings.append(f"{len(const_cols)} constant column(s): {const_cols[:5]}")

    # High-missingness individual columns
    high_miss = [name for name, cp in profiles.items() if cp.missing_pct > 50]
    if high_miss:
        score -= min(10, len(high_miss) * 2)
        warnings.append(f"{len(high_miss)} column(s) with >50% missing values.")

    return round(max(0.0, min(100.0, score)), 1), warnings


def _generate_recommendations(
    df: pd.DataFrame,
    profiles: Dict[str, ColumnProfile],
    numeric_cols: List[str],
    categorical_cols: List[str],
    date_cols: List[str],
    n_rows: int,
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []

    # Correlation matrix
    if len(numeric_cols) >= 2:
        recs.append({
            "id": "rec_correlation_matrix",
            "title": "Correlation Matrix",
            "why": f"{len(numeric_cols)} numeric columns detected. Identify linear relationships.",
            "columns": numeric_cols[:10],
            "method": "Pearson correlation heatmap",
            "output": "Correlation matrix with heatmap",
            "limitations": "Pearson measures linear association only. Outliers can distort results.",
            "category": "relationship",
        })

    # Time-series trend
    if date_cols and numeric_cols:
        recs.append({
            "id": "rec_time_series",
            "title": "Time-Series Trend",
            "why": f"Date column '{date_cols[0]}' and {len(numeric_cols)} numeric column(s) found.",
            "columns": [date_cols[0]] + numeric_cols[:3],
            "method": "Aggregated time series line chart",
            "output": "Trend chart by period",
            "limitations": "Requires sufficient time periods for meaningful trend detection.",
            "category": "time",
        })

    # Grouped comparison
    if categorical_cols and numeric_cols:
        recs.append({
            "id": "rec_group_comparison",
            "title": "Group Comparison",
            "why": f"Category '{categorical_cols[0]}' vs numeric '{numeric_cols[0]}' — compare distributions.",
            "columns": [categorical_cols[0], numeric_cols[0]],
            "method": "Box plot and grouped statistics",
            "output": "Group means, medians, and distribution chart",
            "limitations": "Statistical tests require sample size and normality checks.",
            "category": "relationship",
        })

    # Missing value investigation
    high_miss = {n: p.missing_pct for n, p in profiles.items() if p.missing_pct > 10}
    if high_miss:
        recs.append({
            "id": "rec_missing_analysis",
            "title": "Missing Value Investigation",
            "why": f"{len(high_miss)} column(s) have >10% missing values.",
            "columns": list(high_miss.keys()),
            "method": "Missing value heatmap and pattern analysis",
            "output": "Missingness report with patterns",
            "limitations": "Cannot determine cause of missingness from data alone.",
            "category": "quality",
        })

    # Outlier detection
    if numeric_cols:
        recs.append({
            "id": "rec_outlier_detection",
            "title": "Outlier Detection",
            "why": f"{len(numeric_cols)} numeric columns available for IQR-based outlier flagging.",
            "columns": numeric_cols[:5],
            "method": "IQR rule (1.5×IQR beyond Q1/Q3)",
            "output": "Flagged observations with box plots",
            "limitations": "IQR method may flag genuine extreme values. Investigate before removing.",
            "category": "quality",
        })

    # Category frequency
    if categorical_cols:
        recs.append({
            "id": "rec_category_frequency",
            "title": "Category Frequency Analysis",
            "why": f"'{categorical_cols[0]}' has {profiles.get(categorical_cols[0]).unique_count if profiles.get(categorical_cols[0]) else '?'} unique values.",
            "columns": [categorical_cols[0]],
            "method": "Frequency bar chart with Pareto option",
            "output": "Ranked category counts and proportions",
            "limitations": "Very high-cardinality categories may need grouping.",
            "category": "distribution",
        })

    return recs


def compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics for a numeric series."""
    clean = series.dropna()
    if clean.empty:
        return {}

    try:
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_count = int(((clean < lower_fence) | (clean > upper_fence)).sum())

        return {
            "count": int(len(clean)),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
            "variance": float(clean.var()),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(clean) * 100, 1) if len(clean) else 0,
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurt()),
            "range": float(clean.max() - clean.min()),
        }
    except Exception as exc:
        logger.warning("Stats error for series: %s", exc)
        return {}


def compute_categorical_stats(series: pd.Series, top_n: int = 20) -> Dict[str, Any]:
    """Frequency and distribution statistics for a categorical series."""
    clean = series.dropna()
    if clean.empty:
        return {}

    counts = clean.value_counts()
    total = len(clean)

    return {
        "total": total,
        "unique": int(clean.nunique()),
        "top_values": counts.head(top_n).to_dict(),
        "top_pcts": (counts.head(top_n) / total * 100).round(1).to_dict(),
        "most_frequent": counts.index[0] if len(counts) else None,
        "most_frequent_count": int(counts.iloc[0]) if len(counts) else 0,
        "least_frequent": counts.index[-1] if len(counts) else None,
        "least_frequent_count": int(counts.iloc[-1]) if len(counts) else 0,
        "concentration_top3_pct": round(counts.head(3).sum() / total * 100, 1) if total else 0,
    }
