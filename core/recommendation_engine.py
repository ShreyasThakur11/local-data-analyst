"""
Recommendation engine with rule-based foundation and contextual-bandit adaptive layer.
Rewards are tracked locally via the learning store.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.profiling import DatasetProfile

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    id: str
    title: str
    why: str
    columns: List[str]
    method: str
    output: str
    limitations: str
    category: str
    priority: float = 1.0
    estimated_usefulness: float = 0.5  # 0–1 from bandit


def generate_recommendations(
    profile: DatasetProfile,
    analytical_profile: str = "General Data Analysis",
    user_prefs: Optional[Dict[str, Any]] = None,
) -> List[Recommendation]:
    """
    Generate a ranked list of recommendations for the current dataset.
    Starts from rule-based logic and adjusts scores using bandit weights when available.
    """
    recs: List[Recommendation] = []
    nc = profile.numeric_cols
    cc = profile.categorical_cols
    dc = profile.date_cols
    n = profile.n_rows

    # ── Rule-based candidates ──────────────────────────────────────────────────

    if len(nc) >= 2:
        recs.append(Recommendation(
            id="correlation_matrix",
            title="Correlation Matrix",
            why=f"{len(nc)} numeric columns — explore pairwise linear relationships.",
            columns=nc[:8],
            method="Pearson / Spearman correlation",
            output="Heatmap with coefficient values",
            limitations="Linear correlation only; outliers can distort results.",
            category="relationship",
            priority=0.9,
        ))

    if dc and nc:
        recs.append(Recommendation(
            id="time_series_trend",
            title="Time-Series Trend",
            why=f"Date column '{dc[0]}' + numeric measure — explore trends over time.",
            columns=[dc[0]] + nc[:2],
            method="Aggregated line chart by period",
            output="Trend chart with period-over-period change",
            limitations="Requires sufficient time periods.",
            category="time",
            priority=0.85,
        ))

    if cc and nc:
        recs.append(Recommendation(
            id="group_comparison",
            title="Group Comparison",
            why=f"'{cc[0]}' (category) vs '{nc[0]}' (numeric) — compare group distributions.",
            columns=[cc[0], nc[0]],
            method="Box plot + ANOVA / Kruskal-Wallis",
            output="Group statistics and significance test",
            limitations="Tests assume adequate sample size per group.",
            category="relationship",
            priority=0.8,
        ))

    if profile.duplicate_rows > 0:
        recs.append(Recommendation(
            id="investigate_duplicates",
            title="Investigate Duplicate Rows",
            why=f"{profile.duplicate_rows:,} fully duplicate rows detected.",
            columns=list(profile.columns.keys())[:5],
            method="Duplicate row inspection",
            output="List of duplicate records",
            limitations="Duplicates may be intentional (e.g., log entries).",
            category="quality",
            priority=0.75,
        ))

    if profile.total_missing_pct > 5:
        high_miss_cols = [
            n for n, p in profile.columns.items() if p.missing_pct > 10
        ]
        if high_miss_cols:
            recs.append(Recommendation(
                id="missing_value_analysis",
                title="Missing Value Analysis",
                why=f"{len(high_miss_cols)} columns have >10% missing values.",
                columns=high_miss_cols[:5],
                method="Missingness heatmap and pattern analysis",
                output="Missingness report",
                limitations="Cannot determine cause of missingness from data alone.",
                category="quality",
                priority=0.7,
            ))

    if nc:
        recs.append(Recommendation(
            id="outlier_detection",
            title="Outlier Detection",
            why=f"{len(nc)} numeric columns — flag statistical extremes.",
            columns=nc[:5],
            method="IQR × 1.5 rule",
            output="Flagged observations with box plots",
            limitations="Flagged values are not automatically errors.",
            category="quality",
            priority=0.65,
        ))

    if cc:
        recs.append(Recommendation(
            id="category_frequency",
            title="Category Frequency Analysis",
            why=f"'{cc[0]}' has {profile.columns.get(cc[0]).unique_count if profile.columns.get(cc[0]) else '?'} categories — explore distribution.",
            columns=[cc[0]],
            method="Frequency bar chart",
            output="Ranked category counts",
            limitations="High-cardinality columns may need grouping.",
            category="distribution",
            priority=0.6,
        ))

    if len(cc) >= 2:
        recs.append(Recommendation(
            id="cross_tabulation",
            title="Cross-Tabulation",
            why=f"Two category columns '{cc[0]}' × '{cc[1]}' — explore associations.",
            columns=[cc[0], cc[1]],
            method="Contingency table + Chi-square test",
            output="Cross-tabulation with row percentages",
            limitations="Requires adequate expected cell counts.",
            category="relationship",
            priority=0.55,
        ))

    # ── Domain-specific additions ──────────────────────────────────────────────

    recs = _apply_domain_profile(recs, analytical_profile, profile)

    # ── Bandit adjustment ──────────────────────────────────────────────────────

    if user_prefs:
        recs = _adjust_with_prefs(recs, user_prefs)

    # Sort by adjusted priority
    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs


def _apply_domain_profile(
    recs: List[Recommendation],
    profile_name: str,
    ds_profile: DatasetProfile,
) -> List[Recommendation]:
    nc = ds_profile.numeric_cols
    cc = ds_profile.categorical_cols
    dc = ds_profile.date_cols

    if "Finance" in profile_name and nc and dc:
        recs.append(Recommendation(
            id="period_variance",
            title="Period-over-Period Variance",
            why="Finance profile: explore revenue/cost changes between periods.",
            columns=dc[:1] + nc[:3],
            method="Period comparison chart",
            output="% change and absolute variance by period",
            limitations="Requires consistent period definitions.",
            category="finance",
            priority=0.88,
        ))

    if ("Operations" in profile_name or "Supply Chain" in profile_name) and nc:
        recs.append(Recommendation(
            id="process_variability",
            title="Process Variability Analysis",
            why="Operations profile: identify variability in key process metrics.",
            columns=nc[:4],
            method="Standard deviation + IQR across groups",
            output="Variability summary table and box plots",
            limitations="Requires clear process metric columns.",
            category="operations",
            priority=0.87,
        ))

    if "Energy" in profile_name and nc:
        recs.append(Recommendation(
            id="efficiency_ratio",
            title="Energy Efficiency Ratio",
            why="Energy profile: compute output per unit energy consumed.",
            columns=nc[:4],
            method="Ratio analysis and trend",
            output="Efficiency trend chart",
            limitations="Requires clearly labelled energy and output columns.",
            category="energy",
            priority=0.88,
        ))

    if "MBA" in profile_name:
        recs.append(Recommendation(
            id="regression_analysis",
            title="Regression Analysis",
            why="MBA profile: model relationships between outcome and explanatory variables.",
            columns=nc[:4],
            method="OLS linear regression",
            output="Coefficients, R², residuals",
            limitations="Assumes linearity, independence, homoscedasticity.",
            category="statistics",
            priority=0.83,
        ))

    return recs


def _adjust_with_prefs(
    recs: List[Recommendation], user_prefs: Dict[str, Any]
) -> List[Recommendation]:
    """
    Adjust recommendation priorities using stored preference scores.
    user_prefs = {rec_id: float score}
    """
    for rec in recs:
        score = user_prefs.get(rec.id)
        if score is not None:
            # Blend original priority with learned score (70/30 rule)
            adjusted = 0.7 * rec.priority + 0.3 * score
            rec.priority = round(adjusted, 4)
            rec.estimated_usefulness = round(float(score), 4)
    return recs
