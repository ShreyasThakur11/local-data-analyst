"""Interactive Dashboard Builder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import (BASE_LAYOUT, bar_chart, box_plot, grouped_bar,
                                 histogram, line_chart, scatter_plot,
                                 time_series_with_pct_change)
from components.metric_cards import active_dataset_guard, kpi_card, section_header
from core.statistics_engine import aggregate_time_series

st.title("📊 Dashboard Builder")
st.caption("Build interactive panels from your data. Filter, aggregate, and compare.")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df_full = rec["df"]
profile = rec["profile"]
nc = profile.numeric_cols
cc = profile.categorical_cols
dc = profile.date_cols

# ── Filter panel ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**📊 Dashboard Filters**")
    filters_applied = {}

    for cat_col in cc[:3]:  # limit sidebar filters
        unique_vals = df_full[cat_col].dropna().unique().tolist()
        if 2 <= len(unique_vals) <= 30:
            selected = st.multiselect(f"{cat_col}", unique_vals, key=f"db_filt_{cat_col}")
            if selected:
                filters_applied[cat_col] = selected

    # Date range filter
    if dc:
        date_col = dc[0]
        try:
            parsed = pd.to_datetime(df_full[date_col], errors="coerce").dropna()
            min_d, max_d = parsed.min().date(), parsed.max().date()
            date_range = st.date_input(f"Date range ({date_col})", value=(min_d, max_d), key="db_date_range")
            if len(date_range) == 2:
                filters_applied["__date__"] = (date_col, date_range)
        except Exception:
            pass

    if st.button("🔄 Reset Filters", key="db_reset_filters"):
        st.rerun()

# Apply filters
df = df_full.copy()
for col, vals in filters_applied.items():
    if col == "__date__":
        date_col, (d_min, d_max) = vals
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[(df[date_col].dt.date >= d_min) & (df[date_col].dt.date <= d_max)]
    else:
        df = df[df[col].isin(vals)]

# ── KPI summary row ───────────────────────────────────────────────────────────
section_header(f"Dataset Summary — {len(df):,} / {len(df_full):,} rows")

kpi_cols = st.columns(min(4, len(nc) + 1))
with kpi_cols[0]:
    kpi_card("Rows (filtered)", f"{len(df):,}", icon="📊")

for i, col_name in enumerate(nc[:3]):
    with kpi_cols[i + 1]:
        total = df[col_name].sum()
        kpi_card(f"Total {col_name}", f"{total:,.0f}", icon="💰")

st.markdown("---")

# ── Panel builder ─────────────────────────────────────────────────────────────
section_header("Build Chart Panel")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    chart_type = st.selectbox("Chart type", [
        "Bar Chart", "Grouped Bar", "Line Chart", "Scatter Plot",
        "Box Plot", "Histogram", "Time Series"
    ], key="db_chart_type")

with col_cfg2:
    x_col = st.selectbox("X / Category", list(df.columns), key="db_x")
    y_col = st.selectbox("Y / Measure", ["(count)"] + nc, key="db_y") if chart_type not in ("Histogram",) else None

with col_cfg3:
    color_col = st.selectbox("Colour by (optional)", ["None"] + cc, key="db_color")
    agg_method = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "min", "max"], key="db_agg")

# ── Generate panel ─────────────────────────────────────────────────────────────
if st.button("➕ Add Panel", key="db_add_panel", type="primary"):
    panel = {
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "color_col": color_col if color_col != "None" else None,
        "agg_method": agg_method,
        "title": f"{agg_method.title()} of {y_col or x_col} by {x_col}",
        "filters": dict(filters_applied),
    }
    if "dashboard_panels" not in st.session_state:
        st.session_state["dashboard_panels"] = []
    st.session_state["dashboard_panels"].append(panel)
    st.success(f"Panel added: {panel['title']}")

# ── Render panels ─────────────────────────────────────────────────────────────
panels = st.session_state.get("dashboard_panels", [])

if panels:
    st.markdown("---")
    section_header(f"Dashboard ({len(panels)} panels)")

    for i, panel in enumerate(panels):
        ct = panel["chart_type"]
        x = panel["x_col"]
        y = panel.get("y_col")
        color = panel.get("color_col")
        agg = panel.get("agg_method", "sum")
        title = panel.get("title", "Chart")

        with st.expander(f"📊 {title}", expanded=True):
            try:
                if ct == "Bar Chart" and y and y != "(count)":
                    fig = grouped_bar(df, x, y, agg=agg, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Bar Chart" and (not y or y == "(count)"):
                    fig = bar_chart(df[x], x, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Grouped Bar" and y and y != "(count)":
                    fig = grouped_bar(df, x, y, agg=agg, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Scatter Plot" and y and y != "(count)":
                    fig = scatter_plot(df, x, y, color_col=color, add_trendline=True, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Box Plot" and y and y != "(count)":
                    fig = box_plot(df, y, group_col=x if x in cc else None, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Histogram" and x in nc:
                    fig = histogram(df[x], x, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                elif ct == "Time Series" and y and y != "(count)":
                    ts_r = aggregate_time_series(df, x, y, freq="M", agg_func=agg)
                    if "error" not in ts_r:
                        fig = time_series_with_pct_change(ts_r["series"], ts_r["pct_change"], title=title)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(ts_r["error"])

                elif ct == "Line Chart" and x in dc and y and y != "(count)":
                    ts_r = aggregate_time_series(df, x, y, freq="M", agg_func=agg)
                    if "error" not in ts_r:
                        fig = line_chart(ts_r["series"], title=title, y_label=y)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(ts_r["error"])

                else:
                    # Fallback: simple frequency bar
                    fig = bar_chart(df[x], x, title=title)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Chart error: {e}")

            col_del = st.columns([4, 1])
            with col_del[1]:
                if st.button("🗑️ Remove", key=f"del_panel_{i}"):
                    st.session_state["dashboard_panels"].pop(i)
                    st.rerun()

    # Export table
    if nc and st.button("📥 Download Filtered Data (CSV)", key="db_download"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download", data=csv, file_name="filtered_data.csv", mime="text/csv")

    if st.button("🗑️ Clear All Panels", key="db_clear"):
        st.session_state["dashboard_panels"] = []
        st.rerun()
else:
    st.info("No panels added yet. Configure a chart above and click '➕ Add Panel'.")
