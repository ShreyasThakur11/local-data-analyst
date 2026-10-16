"""
Insights engine: generates data-grounded, deterministic textual explanations.
Does NOT invent statistics or causal claims.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from core.profiling import DatasetProfile

logger = logging.getLogger(__name__)


def summarise_dataset(profile: DatasetProfile) -> str:
    """Generate a factual plain-English summary of the dataset."""
    lines = [
        f"**Dataset:** {profile.source_name}",
        f"**Shape:** {profile.n_rows:,} rows × {profile.n_cols} columns",
        f"**Memory:** {profile.memory_mb} MB",
        "",
        f"**Column types found:** "
        f"{len(profile.numeric_cols)} numeric, "
        f"{len(profile.categorical_cols)} categorical, "
        f"{len(profile.date_cols)} date, "
        f"{len(profile.text_cols)} text, "
        f"{len(profile.id_cols)} identifier, "
        f"{len(profile.bool_cols)} boolean.",
        "",
        f"**Overall data quality score:** {profile.quality_score}/100",
        f"**Total missing values:** {profile.total_missing:,} ({profile.total_missing_pct}%)",
        f"**Duplicate rows:** {profile.duplicate_rows:,}",
    ]

    if profile.warnings:
        lines.append("\n**⚠️ Quality warnings:**")
        for w in profile.warnings[:5]:
            lines.append(f"- {w}")

    return "\n".join(lines)


def explain_correlation(r: float, col_a: str, col_b: str, n: int, p: float) -> str:
    """Generate a grounded plain-English correlation explanation."""
    sign = "positively" if r > 0 else "negatively"
    strength = _corr_strength(abs(r))

    text = (
        f"**{col_a}** and **{col_b}** are {strength} {sign} correlated "
        f"(r = {r:.3f}, n = {n:,}, p = {p:.4f})."
    )

    if p >= 0.05:
        text += (
            " The p-value does not reach the conventional 0.05 threshold, "
            "so this correlation is not statistically significant at that level. "
            "Consider whether the sample size is adequate."
        )
    else:
        text += " This result is statistically significant at the 0.05 level."

    text += (
        "\n\n**Caution:** Correlation measures linear association, not causation. "
        "A third variable may explain this relationship."
    )

    return text


def explain_group_comparison(
    group_col: str, value_col: str, stats_df: pd.DataFrame, anova: Optional[Dict] = None
) -> str:
    """Explain a grouped comparison result."""
    if stats_df.empty:
        return "No group data available."

    lines = [
        f"**{value_col}** was compared across groups defined by **{group_col}** "
        f"({len(stats_df)} groups).",
        "",
    ]

    top = stats_df.iloc[0]
    bot = stats_df.iloc[-1]
    lines.append(
        f"Highest mean: **{top[group_col]}** ({top['mean']:.2f}). "
        f"Lowest mean: **{bot[group_col]}** ({bot['mean']:.2f}). "
        f"Absolute difference: {top['mean'] - bot['mean']:.2f}."
    )

    if anova:
        sig = anova.get("significant_p05")
        p = anova.get("p_value", 1.0)
        lines.append(
            f"\nOne-way ANOVA: F = {anova.get('f_stat', '?')}, p = {p:.4f}. "
            + ("The difference across groups is statistically significant (p < 0.05)."
               if sig else "The difference does not reach statistical significance (p ≥ 0.05).")
        )
        lines.append(
            "\n_Caveat: ANOVA assumes normality and equal variances. "
            "Check the Kruskal-Wallis result for a non-parametric alternative._"
        )

    return "\n".join(lines)


def explain_time_series(
    date_col: str, value_col: str, freq: str, agg_func: str,
    n_periods: int, overall_change: Optional[float]
) -> str:
    freq_label = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "annual"}.get(freq, freq)
    lines = [
        f"**{value_col}** aggregated by {freq_label} {agg_func.lower()} "
        f"across **{n_periods}** periods (date: **{date_col}**).",
    ]
    if overall_change is not None:
        direction = "increased" if overall_change > 0 else "decreased"
        lines.append(
            f"Overall change from first to last period: {direction} by {abs(overall_change):.1f}%."
        )
    lines.append(
        "\n_This is a descriptive summary. Causal interpretation requires additional context._"
    )
    return "\n".join(lines)


def explain_outlier_detection(col: str, n_outliers: int, n_total: int, method: str) -> str:
    pct = n_outliers / n_total * 100 if n_total else 0
    return (
        f"**{n_outliers}** potential outlier(s) detected in **{col}** "
        f"({pct:.1f}% of {n_total:,} values) using the {method}. "
        "These are statistical extremes — not confirmed errors. "
        "Investigate the flagged values in context before taking action."
    )


def consultant_summary(
    observation: str,
    interpretation: str,
    implications: str,
    next_steps: str,
    caveats: str,
) -> str:
    return f"""### Observation
{observation}

### Interpretation
{interpretation}

### Business Implications
{implications}

### Recommended Next Analysis
{next_steps}

### Caveats
{caveats}
"""


def _corr_strength(r: float) -> str:
    if r >= 0.8:
        return "very strongly"
    if r >= 0.6:
        return "strongly"
    if r >= 0.4:
        return "moderately"
    if r >= 0.2:
        return "weakly"
    return "negligibly"
