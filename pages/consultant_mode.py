"""Consultant Mode — structured, evidence-based business insight generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.metric_cards import active_dataset_guard, section_header
from core.profiling import compute_numeric_stats, compute_categorical_stats
from core.statistics_engine import (grouped_comparison, aggregate_time_series,
                                     pearson_correlation)
from core.insights_engine import consultant_summary

st.title("💼 Consultant Mode")
st.markdown("""
Transform analytical findings into structured, evidence-based business insights.
Every claim is grounded in calculated results from the loaded dataset.
**No numbers are invented. Hypothetical scenarios are clearly labelled.**
""")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df = rec["df"]
profile = rec["profile"]
nc = profile.numeric_cols
cc = profile.categorical_cols
dc = profile.date_cols

# ── Analysis selection ────────────────────────────────────────────────────────
section_header("Select Analysis to Interpret")

analysis_type = st.selectbox("Analysis type", [
    "Descriptive Statistics Summary",
    "Group Performance Comparison",
    "Trend Analysis",
    "Correlation Insight",
    "Data Quality Assessment",
    "Custom Observation",
], key="consul_type")

findings = {}
observation_text = ""
interpretation_text = ""
implications_text = ""
next_steps_text = ""
caveats_text = ""

# ─────────────────────────────────────────────────────────────────────────────
if analysis_type == "Descriptive Statistics Summary":
    col_sel = st.selectbox("Select numeric column", nc, key="consul_desc_col")
    if col_sel and st.button("▶ Generate Insight", key="btn_consul_desc", type="primary"):
        stats = compute_numeric_stats(df[col_sel])
        if stats:
            observation_text = (
                f"**{col_sel}** has {stats['count']:,} non-missing observations "
                f"(dataset: {profile.source_name}). "
                f"Mean: {stats['mean']:,.3g}, Median: {stats['median']:,.3g}, "
                f"Std: {stats['std']:,.3g}. "
                f"Range: {stats['min']:,.3g} to {stats['max']:,.3g}. "
                f"IQR: {stats['iqr']:,.3g}. "
                f"Potential outliers (IQR×1.5): {stats['outlier_count']} ({stats['outlier_pct']}%). "
                f"Skewness: {stats['skewness']:.3f}."
            )
            interpretation_text = (
                f"The distribution of {col_sel} "
                + ("appears approximately symmetric." if abs(stats['skewness']) < 0.5
                   else f"is {'right' if stats['skewness'] > 0 else 'left'}-skewed, "
                   f"suggesting the mean is pulled {'above' if stats['skewness'] > 0 else 'below'} the median. ")
                + f" {stats['outlier_count']} outlier(s) fall beyond the standard IQR fences and warrant investigation."
            )
            implications_text = (
                "Skewed distributions or high variability may indicate process instability, "
                "seasonal effects, or data collection inconsistencies. "
                "The outliers should be investigated to determine whether they represent genuine extremes "
                "or data entry errors before drawing operational conclusions."
            )
            next_steps_text = (
                f"1. Investigate the {stats['outlier_count']} flagged observations in context.\n"
                f"2. Compare {col_sel} across groups if a categorical variable is available.\n"
                f"3. Plot a time-series trend if a date column is present."
            )
            caveats_text = (
                f"Statistics based on {stats['count']:,} observations from '{profile.source_name}'. "
                "Outlier flags use IQR×1.5 and should be investigated before removal. "
                "Skewness does not confirm a specific underlying distribution."
            )

# ─────────────────────────────────────────────────────────────────────────────
elif analysis_type == "Group Performance Comparison":
    if not cc or not nc:
        st.warning("Requires at least one categorical and one numeric column.")
        st.stop()
    group_col = st.selectbox("Group column (category)", cc, key="consul_grp_col")
    value_col = st.selectbox("Value column (numeric)", nc, key="consul_grp_val")

    if st.button("▶ Generate Insight", key="btn_consul_grp", type="primary"):
        result = grouped_comparison(df, group_col, value_col)
        if "error" in result:
            st.error(result["error"])
        else:
            stats_df = result["stats"]
            top = stats_df.iloc[0]
            bot = stats_df.iloc[-1]
            gap = top["mean"] - bot["mean"]
            gap_pct = gap / bot["mean"] * 100 if bot["mean"] != 0 else 0

            observation_text = (
                f"**{value_col}** was compared across **{result['n_groups']}** groups "
                f"defined by **{group_col}** ({result['n_valid_rows']:,} valid observations). "
                f"Highest mean: **{top[group_col]}** ({top['mean']:,.3g}). "
                f"Lowest mean: **{bot[group_col]}** ({bot['mean']:,.3g}). "
                f"Absolute gap: {gap:,.3g} ({gap_pct:.1f}%)."
            )

            sig = result.get("anova", {}).get("significant_p05", False)
            interpretation_text = (
                f"The {gap_pct:.1f}% gap between the top and bottom group "
                + ("is statistically significant at the 0.05 level (ANOVA)."
                   if sig else "does not reach statistical significance at the 0.05 level.")
                + " This is an associative observation — not a causal finding."
            )
            implications_text = (
                f"If the difference in {value_col} across {group_col} groups reflects genuine performance variation, "
                "this may represent an opportunity to identify and share best practices from "
                f"**{top[group_col]}** or investigate constraints in **{bot[group_col]}**."
            )
            next_steps_text = (
                f"1. Explore what operational or structural factors differ between {top[group_col]} and {bot[group_col]}.\n"
                f"2. Check whether the gap has narrowed or widened over time (requires date column).\n"
                f"3. Investigate whether the sample size per group is sufficient for reliable conclusions."
            )
            caveats_text = (
                f"Results based on {result['n_valid_rows']:,} rows. "
                "Groups with fewer than 3 observations were excluded. "
                "ANOVA assumes normality and equal variances — see Kruskal-Wallis for a non-parametric alternative. "
                "No causal relationship is established."
            )

# ─────────────────────────────────────────────────────────────────────────────
elif analysis_type == "Trend Analysis":
    if not dc or not nc:
        st.warning("Requires at least one date and one numeric column.")
        st.stop()
    date_col = st.selectbox("Date column", dc, key="consul_ts_date")
    val_col = st.selectbox("Value column", nc, key="consul_ts_val")
    freq = st.selectbox("Period", ["Month", "Quarter", "Year"], key="consul_ts_freq")
    freq_map = {"Month": "M", "Quarter": "Q", "Year": "Y"}

    if st.button("▶ Generate Insight", key="btn_consul_ts", type="primary"):
        ts_result = aggregate_time_series(df, date_col, val_col, freq=freq_map[freq])
        if "error" in ts_result:
            st.error(ts_result["error"])
        else:
            oc = ts_result.get("overall_change_pct")
            direction = "increased" if (oc or 0) > 0 else "decreased"
            ts = ts_result["series"]
            peak_period = ts.idxmax()
            trough_period = ts.idxmin()

            observation_text = (
                f"**{val_col}** aggregated by {freq.lower()} (sum) over {ts_result['n_periods']} periods "
                f"({ts_result['date_range'][0]} to {ts_result['date_range'][1]}). "
                + (f"Overall change: {direction} by {abs(oc):.1f}% from first to last period. " if oc else "")
                + f"Peak period: {peak_period} ({ts.max():,.3g}). "
                f"Trough period: {trough_period} ({ts.min():,.3g})."
            )
            interpretation_text = (
                f"The {('upward' if (oc or 0) > 0 else 'downward')} trend in {val_col} "
                "may reflect organic growth, seasonal patterns, or structural changes in operations. "
                "The peak and trough periods should be investigated for contributing factors."
            )
            implications_text = (
                f"If the trend in {val_col} is sustained, it has implications for planning, "
                "resource allocation, and target-setting. "
                "A declining trend may indicate a performance issue requiring intervention."
            )
            next_steps_text = (
                f"1. Decompose the trend into seasonal and cyclical components if enough periods are available.\n"
                f"2. Overlay a categorical variable (e.g., region, product) to identify drivers of the peak.\n"
                f"3. Compare the trend against an external benchmark if available."
            )
            caveats_text = (
                f"Based on {ts_result['n_periods']} {freq.lower()} periods. "
                "First-to-last overall change does not account for intermediate fluctuations. "
                "Trend interpretation requires domain context."
            )

# ─────────────────────────────────────────────────────────────────────────────
elif analysis_type == "Correlation Insight":
    if len(nc) < 2:
        st.warning("Requires at least 2 numeric columns.")
        st.stop()
    col_a = st.selectbox("Column A", nc, key="consul_corr_a")
    col_b = st.selectbox("Column B", [c for c in nc if c != col_a], key="consul_corr_b")

    if st.button("▶ Generate Insight", key="btn_consul_corr", type="primary"):
        result = pearson_correlation(df[col_a], df[col_b])
        if "error" in result:
            st.error(result["error"])
        else:
            from core.insights_engine import explain_correlation
            observation_text = (
                f"Pearson r = {result['r']:.4f} between **{col_a}** and **{col_b}** "
                f"(n = {result['n']:,}, p = {result['p_value']:.4f}). "
                + ("Statistically significant at p < 0.05." if result['significant_p05']
                   else "Not statistically significant at p < 0.05.")
            )
            interpretation_text = explain_correlation(result['r'], col_a, col_b, result['n'], result['p_value'])
            implications_text = (
                f"A {'significant' if result['significant_p05'] else 'non-significant'} "
                f"{'positive' if result['r'] > 0 else 'negative'} association between {col_a} and {col_b} "
                "may inform how these variables are managed together. "
                "Do not infer causation without a controlled study design."
            )
            next_steps_text = (
                "1. Inspect scatter plot for non-linear patterns or influential outliers.\n"
                "2. Test Spearman correlation for robustness.\n"
                "3. Control for a third variable using partial correlation."
            )
            caveats_text = (
                f"Pearson r assumes a linear relationship. n = {result['n']:,}. "
                "Outliers can inflate or deflate r. "
                "Multiple correlation tests increase false-positive risk — "
                "apply Bonferroni correction if many pairs are tested."
            )

# ─────────────────────────────────────────────────────────────────────────────
elif analysis_type == "Data Quality Assessment":
    from core.data_quality import audit_data_quality
    issues = audit_data_quality(df)
    high = [i for i in issues if i.severity == "high"]
    med = [i for i in issues if i.severity == "medium"]

    observation_text = (
        f"Dataset **{profile.source_name}** contains {profile.n_rows:,} rows and {profile.n_cols} columns. "
        f"Overall quality score: {profile.quality_score}/100. "
        f"Total missing values: {profile.total_missing:,} ({profile.total_missing_pct}%). "
        f"Duplicate rows: {profile.duplicate_rows:,}. "
        f"Quality issues detected: {len(high)} high, {len(med)} medium severity."
    )
    interpretation_text = (
        "Data quality limitations affect the reliability of any analysis performed on this dataset. "
        + (f"High-severity issues include: {'; '.join(i.description for i in high[:3])}. " if high else "")
        + "These should be addressed before drawing conclusions."
    )
    implications_text = (
        "Unreliable or incomplete data can lead to misleading analytical results. "
        "For business decisions, understand the provenance and completeness of key fields "
        "before relying on calculated metrics."
    )
    next_steps_text = (
        "1. Address high-severity issues on the Data Quality page.\n"
        "2. Document all data quality limitations in any report or dashboard.\n"
        "3. Obtain clarification from data owners on the cause of key missing fields."
    )
    caveats_text = (
        "Quality issues may be intentional (e.g., optional fields). "
        "Do not remove records without understanding the reason for missingness."
    )

# ─────────────────────────────────────────────────────────────────────────────
elif analysis_type == "Custom Observation":
    st.markdown("Enter your own analytical observation to structure as a consulting insight.")
    observation_text = st.text_area("Observation (what the data shows, with figures)", key="custom_obs")
    interpretation_text = st.text_area("Interpretation (what it might mean)", key="custom_interp")
    implications_text = st.text_area("Business Implications", key="custom_impl")
    next_steps_text = st.text_area("Recommended Next Analysis", key="custom_next")
    caveats_text = st.text_area("Caveats and Limitations", key="custom_caveat")

# ─────────────────────────────────────────────────────────────────────────────
# Render structured insight
# ─────────────────────────────────────────────────────────────────────────────
if observation_text:
    st.markdown("---")
    section_header("📋 Consultant Mode Output")

    insight_md = consultant_summary(
        observation=observation_text,
        interpretation=interpretation_text,
        implications=implications_text,
        next_steps=next_steps_text,
        caveats=caveats_text,
    )
    st.markdown(insight_md)

    # Export
    profile_label = st.session_state.get("profile", "General")
    export_text = f"""# Consultant Mode Report
**Dataset:** {profile.source_name}
**Analytical Profile:** {profile_label}
**Analysis Type:** {analysis_type}

{insight_md}

---
*Generated by Local Data Analyst. All figures derived from the loaded dataset.
No external data, invented numbers, or unverified claims.*
"""
    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=export_text.encode(),
        file_name=f"consultant_insight_{analysis_type.lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )
