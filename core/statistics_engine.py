"""
Statistics engine: relationship analysis, correlations, group comparisons, time series.
All calculations are exact unless explicitly labelled as sampled.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ── Correlation ────────────────────────────────────────────────────────────────

def pearson_correlation(
    series_a: pd.Series, series_b: pd.Series
) -> Dict[str, Any]:
    clean = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    n = len(clean)
    if n < 3:
        return {"error": "Insufficient observations for correlation (need ≥ 3)."}

    r, p = scipy_stats.pearsonr(clean["a"], clean["b"])
    return {
        "method": "Pearson",
        "r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": n,
        "significant_p05": p < 0.05,
        "interpretation": _interpret_r(r),
        "caveat": "Pearson measures linear association. Non-linear relationships may not be captured.",
    }


def spearman_correlation(
    series_a: pd.Series, series_b: pd.Series
) -> Dict[str, Any]:
    clean = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    n = len(clean)
    if n < 3:
        return {"error": "Insufficient observations (need ≥ 3)."}

    r, p = scipy_stats.spearmanr(clean["a"], clean["b"])
    return {
        "method": "Spearman",
        "rho": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": n,
        "significant_p05": p < 0.05,
        "interpretation": _interpret_r(r),
        "caveat": "Spearman measures monotonic association. Rank-based — robust to outliers.",
    }


def correlation_matrix(
    df: pd.DataFrame, method: str = "pearson", min_n: int = 10
) -> Dict[str, Any]:
    """Pairwise correlation matrix. Returns dict of matrix and pairwise p-values."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"error": "Need ≥ 2 numeric columns."}

    corr_matrix = numeric.corr(method=method)
    n_complete = numeric.notna().all(axis=1).sum()

    pairs = []
    cols = list(numeric.columns)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            pair_data = numeric[[c1, c2]].dropna()
            n = len(pair_data)
            if n < min_n:
                continue
            if method == "pearson":
                r, p = scipy_stats.pearsonr(pair_data[c1], pair_data[c2])
            else:
                r, p = scipy_stats.spearmanr(pair_data[c1], pair_data[c2])
            pairs.append({
                "col1": c1, "col2": c2,
                "r": round(float(r), 3), "p": round(float(p), 4),
                "n": n,
                "abs_r": abs(r),
                "strength": _interpret_r(r),
            })

    pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "matrix": corr_matrix,
        "pairs": pairs,
        "method": method,
        "n_complete_rows": int(n_complete),
        "caveat": (
            "Correlations show association, not causation. "
            "Multiple comparisons increase false-positive risk."
        ),
    }


def _interpret_r(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        label = "very strong"
    elif a >= 0.6:
        label = "strong"
    elif a >= 0.4:
        label = "moderate"
    elif a >= 0.2:
        label = "weak"
    else:
        label = "negligible"
    direction = "positive" if r >= 0 else "negative"
    return f"{label} {direction}"


# ── Grouped comparison ────────────────────────────────────────────────────────

def grouped_comparison(
    df: pd.DataFrame, group_col: str, value_col: str, min_group_size: int = 3
) -> Dict[str, Any]:
    """Compare a numeric variable across categories."""
    if group_col not in df.columns or value_col not in df.columns:
        return {"error": "Column not found."}

    clean = df[[group_col, value_col]].dropna()
    if clean.empty:
        return {"error": "No valid data after removing missing values."}

    groups = clean.groupby(group_col)[value_col]
    stats = groups.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    stats.columns = [group_col, "count", "mean", "median", "std", "min", "max"]
    stats = stats[stats["count"] >= min_group_size].sort_values("mean", ascending=False)

    result: Dict[str, Any] = {
        "group_col": group_col,
        "value_col": value_col,
        "stats": stats,
        "n_groups": len(stats),
        "n_valid_rows": len(clean),
    }

    # Statistical test: ANOVA if ≥ 2 groups with enough observations
    group_data = [
        clean[clean[group_col] == g][value_col].values
        for g in stats[group_col]
        if len(clean[clean[group_col] == g]) >= min_group_size
    ]

    if len(group_data) >= 2:
        try:
            f_stat, p_val = scipy_stats.f_oneway(*group_data)
            result["anova"] = {
                "f_stat": round(float(f_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_p05": p_val < 0.05,
                "caveat": (
                    "One-way ANOVA assumes normality and homogeneity of variance. "
                    "Use Kruskal-Wallis for small or non-normal samples."
                ),
            }
            # Kruskal-Wallis as non-parametric alternative
            h_stat, kw_p = scipy_stats.kruskal(*group_data)
            result["kruskal"] = {
                "h_stat": round(float(h_stat), 4),
                "p_value": round(float(kw_p), 6),
                "significant_p05": kw_p < 0.05,
            }
        except Exception as exc:
            result["test_error"] = str(exc)

    return result


# ── Cross-tabulation ──────────────────────────────────────────────────────────

def cross_tabulation(
    df: pd.DataFrame, col_a: str, col_b: str, max_cats: int = 20
) -> Dict[str, Any]:
    clean = df[[col_a, col_b]].dropna()
    if clean.empty:
        return {"error": "No valid data."}

    # Limit categories
    top_a = clean[col_a].value_counts().head(max_cats).index
    top_b = clean[col_b].value_counts().head(max_cats).index
    clean = clean[clean[col_a].isin(top_a) & clean[col_b].isin(top_b)]

    ct = pd.crosstab(clean[col_a], clean[col_b])
    ct_pct = pd.crosstab(clean[col_a], clean[col_b], normalize="index").round(3) * 100

    result: Dict[str, Any] = {
        "counts": ct,
        "row_pct": ct_pct,
        "col_a": col_a,
        "col_b": col_b,
        "n": len(clean),
    }

    # Chi-square test
    try:
        chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
        result["chi2"] = {
            "statistic": round(float(chi2), 4),
            "p_value": round(float(p), 6),
            "dof": int(dof),
            "significant_p05": p < 0.05,
            "caveat": "Chi-square requires expected cell count ≥ 5 in most cells.",
        }
        # Cramér's V
        n = ct.values.sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n > 0 and min_dim > 0) else 0
        result["cramers_v"] = round(float(cramers_v), 4)
    except Exception as exc:
        result["test_error"] = str(exc)

    return result


# ── Time series ───────────────────────────────────────────────────────────────

def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "M",
    agg_func: str = "sum",
) -> Dict[str, Any]:
    """
    Aggregate a numeric column by time period.
    freq: 'D', 'W', 'M', 'Q', 'Y'
    agg_func: 'sum', 'mean', 'median', 'count'
    """
    clean = df[[date_col, value_col]].copy()

    # Parse dates
    try:
        clean[date_col] = pd.to_datetime(clean[date_col], infer_datetime_format=True, errors="coerce")
    except Exception:
        return {"error": f"Could not parse '{date_col}' as dates."}

    clean = clean.dropna(subset=[date_col])
    if len(clean) < 2:
        return {"error": "Insufficient date observations (need ≥ 2)."}

    clean = clean.set_index(date_col).sort_index()
    
    agg_map = {
        "sum": "sum", "mean": "mean", "median": "median", "count": "count",
        "min": "min", "max": "max",
    }
    if agg_func not in agg_map:
        agg_func = "sum"

    freq_map = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
    pd_freq = freq_map.get(freq, "ME")

    try:
        ts = clean[value_col].resample(pd_freq).agg(agg_map[agg_func]).dropna()
    except Exception as exc:
        return {"error": str(exc)}

    if len(ts) < 2:
        return {"error": "Too few periods after aggregation."}

    # Period-over-period change
    pct_change = ts.pct_change() * 100

    return {
        "series": ts,
        "pct_change": pct_change,
        "freq": freq,
        "agg_func": agg_func,
        "n_periods": len(ts),
        "date_range": (ts.index.min(), ts.index.max()),
        "overall_change_pct": round(
            (float(ts.iloc[-1]) - float(ts.iloc[0])) / abs(float(ts.iloc[0])) * 100, 1
        ) if ts.iloc[0] != 0 else None,
    }


# ── Simple linear regression ──────────────────────────────────────────────────

def simple_regression(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Dict[str, Any]:
    """OLS simple linear regression."""
    clean = df[[x_col, y_col]].dropna()
    n = len(clean)

    if n < 5:
        return {"error": "Need ≥ 5 observations for regression."}

    try:
        slope, intercept, r, p, se = scipy_stats.linregress(clean[x_col], clean[y_col])
        r_squared = r ** 2
        return {
            "x_col": x_col,
            "y_col": y_col,
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 6),
            "r": round(float(r), 4),
            "r_squared": round(float(r_squared), 4),
            "p_value": round(float(p), 6),
            "std_error": round(float(se), 6),
            "n": n,
            "significant_p05": p < 0.05,
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "caveat": (
                "OLS assumes linearity, constant variance, and independence. "
                "R² applies only within the range of training data."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Outlier detection ─────────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> Dict[str, Any]:
    """Flag outliers using IQR rule."""
    clean = series.dropna()
    if clean.empty:
        return {}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask_lower = clean < lower
    mask_upper = clean > upper
    outlier_mask = mask_lower | mask_upper

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "n_outliers": int(outlier_mask.sum()),
        "n_low": int(mask_lower.sum()),
        "n_high": int(mask_upper.sum()),
        "outlier_pct": round(outlier_mask.mean() * 100, 1),
        "outlier_indices": list(clean[outlier_mask].index),
        "method": f"IQR × {k}",
        "caveat": "Flagged values are statistical extremes, not confirmed errors.",
    }
