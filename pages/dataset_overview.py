"""Dataset Overview page — KPI cards, profiling summary, recommendations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from components.metric_cards import active_dataset_guard, kpi_card, section_header, recommendation_card, render_data_dictionary
from components.charts import quality_score_gauge, missing_heatmap
from core.recommendation_engine import generate_recommendations
from learning.preference_store import get_store

st.title("📋 Dataset Overview")

rec = active_dataset_guard()
if rec is None:
    st.stop()

profile = rec["profile"]
df = rec["df"]

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Rows", f"{profile.n_rows:,}", icon="📊")
with c2:
    kpi_card("Columns", str(profile.n_cols), icon="📐")
with c3:
    kpi_card("Missing Values", f"{profile.total_missing:,}", delta=f"{profile.total_missing_pct}%", icon="❓")
with c4:
    kpi_card("Duplicate Rows", f"{profile.duplicate_rows:,}", icon="🔄")
with c5:
    kpi_card("Memory", f"{profile.memory_mb} MB", icon="💾")
with c6:
    kpi_card("Quality", f"{profile.quality_score}/100", icon="🩺")

st.markdown("---")

# ── Left column: quality + missing ────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    section_header("Data Quality Score")
    st.plotly_chart(quality_score_gauge(profile.quality_score), use_container_width=True)

    section_header("Column Types")
    type_data = {
        "Numeric": len(profile.numeric_cols),
        "Categorical": len(profile.categorical_cols),
        "Date": len(profile.date_cols),
        "Text": len(profile.text_cols),
        "Identifier": len(profile.id_cols),
        "Boolean": len(profile.bool_cols),
        "Other": profile.n_cols - sum([
            len(profile.numeric_cols), len(profile.categorical_cols),
            len(profile.date_cols), len(profile.text_cols),
            len(profile.id_cols), len(profile.bool_cols),
        ]),
    }
    for k, v in type_data.items():
        if v:
            st.markdown(f"**{k}:** {v}")

with col_right:
    section_header("Missing Values by Column")
    st.plotly_chart(missing_heatmap(df), use_container_width=True)

st.markdown("---")

# ── Data Dictionary ───────────────────────────────────────────────────────────
section_header("Data Dictionary")
col_roles = rec.get("column_roles", {})
render_data_dictionary(profile.columns, overrides=col_roles)

# Export data dictionary
dd_rows = []
for col_name, cp in profile.columns.items():
    dd_rows.append({
        "Column Name": col_name,
        "Original Type": cp.original_type,
        "Inferred Type": cp.inferred_type.value,
        "Semantic Role": col_roles.get(col_name, cp.semantic_role.value),
        "Non-Missing Count": cp.non_missing,
        "Missing Count": cp.missing_count,
        "Missing %": cp.missing_pct,
        "Unique Values": cp.unique_count,
        "Example Values": ", ".join(str(v) for v in cp.example_values[:3]),
        "Suggested Analyses": "; ".join(cp.suggested_analyses[:2]),
    })

import pandas as pd
dd_df = pd.DataFrame(dd_rows)
csv_bytes = dd_df.to_csv(index=False).encode()
st.download_button(
    "⬇️ Download Data Dictionary (CSV)",
    data=csv_bytes,
    file_name=f"data_dictionary_{rec['name']}.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Warnings ──────────────────────────────────────────────────────────────────
if profile.warnings:
    section_header("⚠️ Dataset Warnings")
    for w in profile.warnings:
        st.warning(w)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown("---")
section_header("💡 Recommended Analyses")
st.caption("Based on the dataset structure. Run or dismiss each recommendation.")

store = get_store()
bandit_scores = store.get_bandit_scores()

recs = generate_recommendations(
    profile,
    analytical_profile=st.session_state.get("profile", "General Data Analysis"),
    user_prefs=bandit_scores,
)

if not recs:
    st.info("No recommendations available for this dataset structure.")
else:
    for r in recs[:6]:
        store.record_shown(r.id)
        with st.expander(f"💡 {r.title} — Priority: {int(r.priority*100)}%", expanded=False):
            st.markdown(f"**Why:** {r.why}")
            st.markdown(f"**Method:** {r.method}")
            st.markdown(f"**Output:** {r.output}")
            st.markdown(f"**Columns:** `{'`, `'.join(r.columns)}`")
            st.caption(f"⚠️ {r.limitations}")

            feedback_cols = st.columns(4)
            for i, (label, rating) in enumerate([
                ("👍 Useful", "useful"), ("👎 Not useful", "not_useful"),
                ("⚠️ Incorrect", "incorrect"), ("⭐ Save", "saved"),
            ]):
                if feedback_cols[i].button(label, key=f"fb_{r.id}_{rating}"):
                    store.record_feedback(r.id, rating)
                    st.success(f"Feedback recorded: {label}")
