"""Column Explorer page — per-column deep-dive with statistics and visualisations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (bar_chart, box_plot, histogram,
                                 scatter_plot, line_chart)
from components.metric_cards import active_dataset_guard, section_header, type_badge
from core.profiling import compute_categorical_stats, compute_numeric_stats
from core.statistics_engine import detect_outliers_iqr
from core.type_inference import InferredType

st.title("🔍 Column Explorer")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df = rec["df"]
profile = rec["profile"]

# ── Column selector ────────────────────────────────────────────────────────────
col_names = list(df.columns)
selected_col = st.selectbox("Select column to explore", col_names, key="col_explorer_select")

if not selected_col:
    st.stop()

cp = profile.columns.get(selected_col)
series = df[selected_col]

# ── Header ────────────────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"### `{selected_col}`")
    if cp:
        st.markdown(cp.inferred_type.value, unsafe_allow_html=True)
with col_b:
    st.metric("Non-Missing", f"{cp.non_missing:,}" if cp else "–")
with col_c:
    st.metric("Missing", f"{cp.missing_count:,} ({cp.missing_pct:.1f}%)" if cp else "–")
with col_d:
    st.metric("Unique Values", f"{cp.unique_count:,}" if cp else "–")

if cp and cp.notes:
    for note in cp.notes:
        st.info(note)

st.markdown("---")

# ── Inferred type and manual override ─────────────────────────────────────────
section_header("Type & Role")
col_type, col_role = st.columns(2)

type_options = [t.value for t in InferredType]
role_options = [
    "Identifier", "Date / Timestamp", "Category / Dimension", "Numeric Measure",
    "Outcome / Target", "Grouping Variable", "Weight", "Geography", "Free Text",
    "Currency", "Percentage", "Duration", "Quantity", "Boolean", "Exclude from Analysis", "Unknown"
]

current_type = cp.inferred_type.value if cp else type_options[0]
current_role = rec.get("column_roles", {}).get(selected_col, cp.semantic_role.value if cp else role_options[-1])

with col_type:
    new_type = st.selectbox("Inferred Type", type_options,
                            index=type_options.index(current_type) if current_type in type_options else 0,
                            key="col_type_override")
with col_role:
    new_role = st.selectbox("Semantic Role", role_options,
                            index=role_options.index(current_role) if current_role in role_options else 0,
                            key="col_role_override")

if st.button("💾 Save Classification", key="save_classification"):
    if "column_roles" not in rec:
        rec["column_roles"] = {}
    rec["column_roles"][selected_col] = new_role
    st.session_state["workspace"][st.session_state["active_dataset_id"]]["column_roles"] = rec["column_roles"]
    st.success(f"Saved: `{selected_col}` → {new_role}")

st.markdown("---")

# ── Analysis by type ─────────────────────────────────────────────────────────
is_numeric = pd.api.types.is_numeric_dtype(series)
is_datetime = pd.api.types.is_datetime64_any_dtype(series)
n_unique = series.nunique()

if is_numeric and n_unique > 2:
    section_header("Descriptive Statistics")
    stats = compute_numeric_stats(series)
    if stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{stats['mean']:,.4g}")
        s2.metric("Median", f"{stats['median']:,.4g}")
        s3.metric("Std Dev", f"{stats['std']:,.4g}")
        s4.metric("IQR", f"{stats['iqr']:,.4g}")

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Min", f"{stats['min']:,.4g}")
        s6.metric("Max", f"{stats['max']:,.4g}")
        s7.metric("Skewness", f"{stats['skewness']:.3f}")
        s8.metric("Outliers (IQR×1.5)", f"{stats['outlier_count']} ({stats['outlier_pct']}%)")

    tab_hist, tab_box = st.tabs(["Histogram", "Box Plot"])
    with tab_hist:
        nbins = st.slider("Bins", 10, 100, 30, key="col_hist_bins")
        st.plotly_chart(histogram(series, selected_col, nbins=nbins), use_container_width=True)
    with tab_box:
        # Allow optional grouping
        cat_cols = profile.categorical_cols
        group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="col_box_group")
        group_col = None if group_by == "None" else group_by
        st.plotly_chart(box_plot(df, selected_col, group_col=group_col), use_container_width=True)

    # Outlier details
    outliers = detect_outliers_iqr(series)
    if outliers.get("n_outliers", 0) > 0:
        with st.expander(f"🔍 View {outliers['n_outliers']} potential outliers"):
            outlier_idx = outliers.get("outlier_indices", [])[:50]
            st.dataframe(df.loc[outlier_idx, [selected_col]], use_container_width=True)
            st.caption(f"IQR fences: [{outliers['lower_fence']:.3g}, {outliers['upper_fence']:.3g}]")

elif is_datetime:
    section_header("Date / Time Analysis")
    clean_dates = pd.to_datetime(series, errors="coerce").dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Earliest", str(clean_dates.min().date()))
    col2.metric("Latest", str(clean_dates.max().date()))
    col3.metric("Span", f"{(clean_dates.max() - clean_dates.min()).days} days")

    freq_opt = st.selectbox("Aggregate by", ["Month", "Quarter", "Year", "Week", "Day"], key="date_freq")
    freq_map = {"Month": "ME", "Quarter": "QE", "Year": "YE", "Week": "W", "Day": "D"}
    counts = clean_dates.dt.to_period(freq_map.get(freq_opt, "ME")[0]).value_counts().sort_index()
    ts = pd.Series(counts.values, index=counts.index.astype(str))
    st.plotly_chart(line_chart(ts, title=f"Record Count by {freq_opt}", y_label="Count"), use_container_width=True)

else:
    section_header("Categorical Analysis")
    stats = compute_categorical_stats(series)
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total (non-null)", f"{stats['total']:,}")
        c2.metric("Unique Categories", f"{stats['unique']:,}")
        c3.metric("Most Frequent", f"{stats['most_frequent']} ({stats['most_frequent_count']:,})")

    top_n = st.slider("Top N categories", 5, 50, 20, key="col_cat_topn")
    orient = st.radio("Orientation", ["Horizontal", "Vertical"], key="col_bar_orient", horizontal=True)
    st.plotly_chart(
        bar_chart(series, selected_col, top_n=top_n, orientation="h" if orient == "Horizontal" else "v"),
        use_container_width=True
    )

    # Value table
    with st.expander("View frequency table"):
        vc = series.value_counts().head(top_n).reset_index()
        vc.columns = [selected_col, "Count"]
        vc["Percentage"] = (vc["Count"] / len(series.dropna()) * 100).round(1)
        st.dataframe(vc, use_container_width=True, hide_index=True)

# ── Suggested analyses ────────────────────────────────────────────────────────
if cp and cp.suggested_analyses:
    section_header("Suggested Analyses")
    for s in cp.suggested_analyses:
        st.markdown(f"- {s}")
