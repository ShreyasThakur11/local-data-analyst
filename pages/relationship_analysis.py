"""Relationship Analysis page — correlations, group comparisons, time series, cross-tabs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.charts import (bar_chart, box_plot, correlation_heatmap,
                                 grouped_bar, line_chart, regression_plot,
                                 scatter_plot, time_series_with_pct_change)
from components.metric_cards import active_dataset_guard, section_header
from core.statistics_engine import (aggregate_time_series, correlation_matrix,
                                     cross_tabulation, grouped_comparison,
                                     pearson_correlation, simple_regression,
                                     spearman_correlation)
from core.insights_engine import (explain_correlation, explain_group_comparison,
                                   explain_time_series)

st.title("📈 Relationship Analysis")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df = rec["df"]
profile = rec["profile"]

nc = profile.numeric_cols
cc = profile.categorical_cols
dc = profile.date_cols
all_cols = list(df.columns)

# ── Analysis type tabs ────────────────────────────────────────────────────────
tab_corr, tab_num_vs_num, tab_cat_vs_num, tab_cat_vs_cat, tab_time, tab_multi = st.tabs([
    "🔗 Correlation Matrix", "📉 Numeric × Numeric",
    "📊 Category × Numeric", "🔀 Category × Category",
    "📅 Time Series", "🌐 Multi-Variable"
])

# ─────────────────────────────────────────────────────────────────────────────
# Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    section_header("Correlation Matrix")
    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns for a correlation matrix.")
        st.stop()

    selected_num = st.multiselect(
        "Select numeric columns (2–15 recommended)",
        nc, default=nc[:min(8, len(nc))], key="corr_cols"
    )
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_method")

    if len(selected_num) < 2:
        st.info("Select at least 2 numeric columns.")
    else:
        with st.spinner("Computing correlations…"):
            result = correlation_matrix(df[selected_num], method=method)

        if "error" in result:
            st.error(result["error"])
        else:
            st.plotly_chart(correlation_heatmap(result["matrix"], title=f"{method.title()} Correlation Matrix"), use_container_width=True)
            st.caption(f"⚠️ {result['caveat']}")

            st.markdown("**Strongest pairwise relationships:**")
            pairs_df = pd.DataFrame(result["pairs"])
            if not pairs_df.empty:
                st.dataframe(
                    pairs_df[["col1", "col2", "r", "p", "n", "strength"]].head(10).rename(columns={
                        "col1": "Column A", "col2": "Column B", "r": "Correlation",
                        "p": "p-value", "n": "N", "strength": "Strength"
                    }),
                    use_container_width=True, hide_index=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# Numeric × Numeric
# ─────────────────────────────────────────────────────────────────────────────
with tab_num_vs_num:
    section_header("Numeric vs Numeric")
    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns.")
        st.stop()

    col_x = st.selectbox("X axis (independent)", nc, key="nn_x")
    col_y = st.selectbox("Y axis (dependent)", [c for c in nc if c != col_x], key="nn_y")
    col_color = st.selectbox("Colour by (optional)", ["None"] + cc, key="nn_color")
    add_trend = st.checkbox("Add trend line", value=True, key="nn_trend")
    add_reg = st.checkbox("Show regression statistics", value=True, key="nn_reg")

    color_col = None if col_color == "None" else col_color

    st.plotly_chart(
        scatter_plot(df, col_x, col_y, color_col=color_col, add_trendline=add_trend),
        use_container_width=True
    )

    if add_reg:
        with st.spinner("Running regression…"):
            r_pearson = pearson_correlation(df[col_x], df[col_y])
            reg = simple_regression(df, col_x, col_y)

        if "error" not in r_pearson:
            st.markdown(explain_correlation(r_pearson["r"], col_x, col_y, r_pearson["n"], r_pearson["p_value"]))

        if "error" not in reg:
            st.markdown(f"**Regression equation:** `{reg['equation']}`")
            cols = st.columns(4)
            cols[0].metric("R²", f"{reg['r_squared']:.4f}")
            cols[1].metric("Slope", f"{reg['slope']:.4f}")
            cols[2].metric("Intercept", f"{reg['intercept']:.4f}")
            cols[3].metric("p-value", f"{reg['p_value']:.4f}")
            st.caption(f"⚠️ {reg['caveat']}")

            st.plotly_chart(
                regression_plot(df, col_x, col_y, reg["slope"], reg["intercept"], reg["r_squared"]),
                use_container_width=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# Category × Numeric
# ─────────────────────────────────────────────────────────────────────────────
with tab_cat_vs_num:
    section_header("Category vs Numeric")
    if not cc or not nc:
        st.warning("Need at least one categorical and one numeric column.")
        st.stop()

    cat_col = st.selectbox("Category (group) column", cc, key="cn_cat")
    num_col = st.selectbox("Numeric (value) column", nc, key="cn_num")
    top_cats = st.slider("Max categories to show", 5, 30, 15, key="cn_topn")

    # Filter to top categories
    top_cat_vals = df[cat_col].value_counts().head(top_cats).index
    df_filt = df[df[cat_col].isin(top_cat_vals)]

    with st.spinner("Computing group statistics…"):
        grp = grouped_comparison(df_filt, cat_col, num_col)

    if "error" in grp:
        st.error(grp["error"])
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                grp["stats"].round(2).rename(columns={cat_col: "Group"}),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.plotly_chart(box_plot(df_filt, num_col, group_col=cat_col), use_container_width=True)

        if "anova" in grp:
            anova = grp["anova"]
            st.markdown(explain_group_comparison(cat_col, num_col, grp["stats"], anova))

        st.plotly_chart(grouped_bar(df_filt, cat_col, num_col, top_n=top_cats), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Category × Category
# ─────────────────────────────────────────────────────────────────────────────
with tab_cat_vs_cat:
    section_header("Category vs Category")
    if len(cc) < 2:
        st.warning("Need at least 2 categorical columns.")
        st.stop()

    cat_a = st.selectbox("Category A (rows)", cc, key="cc_a")
    cat_b = st.selectbox("Category B (columns)", [c for c in cc if c != cat_a], key="cc_b")
    max_cats = st.slider("Max categories per column", 5, 30, 10, key="cc_max")

    with st.spinner("Computing cross-tabulation…"):
        ct_result = cross_tabulation(df, cat_a, cat_b, max_cats=max_cats)

    if "error" in ct_result:
        st.error(ct_result["error"])
    else:
        st.markdown("**Count table:**")
        st.dataframe(ct_result["counts"], use_container_width=True)
        st.markdown("**Row percentages (%):**")
        st.dataframe(ct_result["row_pct"].round(1), use_container_width=True)

        if "chi2" in ct_result:
            chi2 = ct_result["chi2"]
            sig = "✅ Significant" if chi2["significant_p05"] else "❌ Not significant"
            st.markdown(
                f"**Chi-square:** {chi2['statistic']:.3f} | **p-value:** {chi2['p_value']:.4f} | "
                f"**{sig}** at p < 0.05 | **Cramér's V:** {ct_result.get('cramers_v', '?'):.3f}"
            )
            st.caption(f"⚠️ {chi2['caveat']}")

# ─────────────────────────────────────────────────────────────────────────────
# Time Series
# ─────────────────────────────────────────────────────────────────────────────
with tab_time:
    section_header("Time Series Analysis")
    if not dc:
        st.warning("No date columns detected. Go to Column Explorer to reclassify a column as Date.")
        st.stop()
    if not nc:
        st.warning("No numeric columns available to trend.")
        st.stop()

    date_col = st.selectbox("Date column", dc, key="ts_date")
    val_col = st.selectbox("Value column", nc, key="ts_val")
    freq = st.selectbox("Aggregation period", ["Month", "Quarter", "Year", "Week", "Day"], key="ts_freq")
    agg = st.selectbox("Aggregation method", ["sum", "mean", "median", "count", "min", "max"], key="ts_agg")

    freq_map = {"Month": "M", "Quarter": "Q", "Year": "Y", "Week": "W", "Day": "D"}

    with st.spinner("Aggregating time series…"):
        ts_result = aggregate_time_series(df, date_col, val_col, freq=freq_map[freq], agg_func=agg)

    if "error" in ts_result:
        st.error(ts_result["error"])
    else:
        st.plotly_chart(
            time_series_with_pct_change(ts_result["series"], ts_result["pct_change"],
                                        title=f"{agg.title()} of {val_col} by {freq}"),
            use_container_width=True
        )
        oc = ts_result.get("overall_change_pct")
        st.markdown(explain_time_series(date_col, val_col, freq_map[freq], agg,
                                       ts_result["n_periods"], oc))

# ─────────────────────────────────────────────────────────────────────────────
# Multi-variable
# ─────────────────────────────────────────────────────────────────────────────
with tab_multi:
    section_header("Multi-Variable Analysis")
    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns.")
        st.stop()

    selected_mv = st.multiselect("Select numeric columns", nc, default=nc[:min(5, len(nc))], key="mv_cols")
    outcome_col = st.selectbox("Outcome / target column (optional)", ["None"] + nc, key="mv_outcome")

    if len(selected_mv) >= 2:
        with st.spinner("Computing pairwise correlations…"):
            result = correlation_matrix(df[selected_mv])

        if "error" not in result:
            st.plotly_chart(
                correlation_heatmap(result["matrix"], title="Pairwise Correlation Matrix"),
                use_container_width=True
            )

    if outcome_col != "None" and outcome_col in df.columns:
        st.markdown(f"**Correlation of all numeric columns with `{outcome_col}`:**")
        corr_rows = []
        for col in nc:
            if col == outcome_col:
                continue
            pair = pd.DataFrame({"a": df[col], "b": df[outcome_col]}).dropna()
            if len(pair) < 5:
                continue
            from scipy.stats import pearsonr
            r, p = pearsonr(pair["a"], pair["b"])
            corr_rows.append({"Column": col, "Correlation (r)": round(r, 3), "p-value": round(p, 4),
                              "Significant (p<0.05)": p < 0.05})
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows).sort_values("Correlation (r)", key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
