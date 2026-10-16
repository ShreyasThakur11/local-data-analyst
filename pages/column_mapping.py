"""Column Mapping page — assign semantic roles and configure analysis selections."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.metric_cards import active_dataset_guard, section_header
from core.type_inference import SemanticRole, InferredType

st.title("🗂️ Column Mapping")
st.markdown("Assign semantic roles to columns and configure variable selection for analysis.")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df = rec["df"]
profile = rec["profile"]

if "column_roles" not in rec:
    rec["column_roles"] = {}

# ── Bulk role assignment ──────────────────────────────────────────────────────
section_header("Column Role Assignment")
st.caption("Override the automatically inferred roles. Changes are used across all analysis pages.")

ROLE_OPTIONS = [r.value for r in SemanticRole]

# Build editable table
role_rows = []
for col_name, cp in profile.columns.items():
    auto_role = cp.semantic_role.value
    user_role = rec["column_roles"].get(col_name, auto_role)
    role_rows.append({
        "Column": col_name,
        "Inferred Type": cp.inferred_type.value,
        "Auto Role": auto_role,
        "Your Role": user_role,
        "Missing %": cp.missing_pct,
        "Unique Values": cp.unique_count,
    })

role_df = pd.DataFrame(role_rows)

# Allow per-column editing
st.markdown("**Edit roles below:**")
col_to_edit = st.selectbox("Select column to change role", list(df.columns), key="role_col_select")
cp_edit = profile.columns.get(col_to_edit)
current_role = rec["column_roles"].get(col_to_edit, cp_edit.semantic_role.value if cp_edit else "Unknown")

new_role = st.selectbox(
    f"Role for `{col_to_edit}`",
    ROLE_OPTIONS,
    index=ROLE_OPTIONS.index(current_role) if current_role in ROLE_OPTIONS else 0,
    key="role_new_value"
)

if st.button("💾 Save Role", key="save_role_btn", type="primary"):
    rec["column_roles"][col_to_edit] = new_role
    st.session_state["workspace"][st.session_state["active_dataset_id"]]["column_roles"] = rec["column_roles"]
    st.success(f"✅ `{col_to_edit}` → **{new_role}**")

# Show current role table
st.markdown("**Current role assignments:**")
display_df = pd.DataFrame([
    {
        "Column": col,
        "Auto Role": profile.columns[col].semantic_role.value if col in profile.columns else "?",
        "Applied Role": rec["column_roles"].get(col, profile.columns[col].semantic_role.value if col in profile.columns else "?"),
        "Overridden": col in rec["column_roles"],
    }
    for col in df.columns
])
st.dataframe(display_df, use_container_width=True, hide_index=True)

if st.button("🔄 Reset All Roles to Auto", key="reset_roles"):
    rec["column_roles"] = {}
    st.session_state["workspace"][st.session_state["active_dataset_id"]]["column_roles"] = {}
    st.success("All roles reset to automatic inference.")
    st.rerun()

st.markdown("---")

# ── Analysis variable selector ────────────────────────────────────────────────
section_header("Analysis Variable Configuration")
st.caption("Configure variables for specific analysis types. These selections drive Relationship Analysis.")

# Get columns by effective role
def cols_by_role(role_val: str) -> list:
    result = []
    for col in df.columns:
        effective = rec["column_roles"].get(col) or (
            profile.columns[col].semantic_role.value if col in profile.columns else ""
        )
        if effective == role_val:
            result.append(col)
    return result

numeric_cols = (
    cols_by_role("Numeric Measure") +
    cols_by_role("Currency") +
    cols_by_role("Percentage") +
    cols_by_role("Quantity") +
    profile.numeric_cols
)
numeric_cols = list(dict.fromkeys(numeric_cols))  # deduplicate preserving order

cat_cols = cols_by_role("Category / Dimension") or profile.categorical_cols
date_cols = cols_by_role("Date / Timestamp") or profile.date_cols

# Analysis config form
with st.form("analysis_config_form"):
    st.markdown("**Select variables for the main analyses:**")
    fc1, fc2 = st.columns(2)
    with fc1:
        x_var = st.selectbox("X variable", ["(none)"] + list(df.columns), key="form_x")
        y_var = st.selectbox("Y variable", ["(none)"] + list(df.columns), key="form_y")
        outcome_var = st.selectbox("Outcome / target", ["(none)"] + numeric_cols, key="form_outcome")
    with fc2:
        date_var = st.selectbox("Date variable", ["(none)"] + date_cols, key="form_date")
        group_var = st.selectbox("Group-by variable", ["(none)"] + cat_cols, key="form_group")
        filter_cols = st.multiselect("Filter columns", list(df.columns), key="form_filters")

    agg_methods = ["sum", "mean", "median", "count", "min", "max", "distinct count"]
    agg_method = st.selectbox("Default aggregation", agg_methods, key="form_agg")

    submitted = st.form_submit_button("💾 Save Analysis Configuration")
    if submitted:
        config = {
            "x_var": x_var if x_var != "(none)" else None,
            "y_var": y_var if y_var != "(none)" else None,
            "outcome_var": outcome_var if outcome_var != "(none)" else None,
            "date_var": date_var if date_var != "(none)" else None,
            "group_var": group_var if group_var != "(none)" else None,
            "filter_cols": filter_cols,
            "agg_method": agg_method,
        }
        st.session_state["workspace"][st.session_state["active_dataset_id"]]["analysis_config"] = config
        st.success("Analysis configuration saved.")

# ── Valid combination guidance ─────────────────────────────────────────────────
st.markdown("---")
section_header("Column Combination Guide")
st.markdown("""
| X Type | Y Type | Recommended Method |
|--------|--------|-------------------|
| Numeric | Numeric | Scatter plot, Pearson/Spearman correlation, OLS regression |
| Category | Numeric | Box plot, grouped statistics, ANOVA / Kruskal-Wallis |
| Date | Numeric | Time-series trend, period-over-period change |
| Category | Category | Cross-tabulation, Chi-square, Cramér's V |
| Identifier | Measure | Record-level view only (avoid aggregating IDs) |
| Text | Any | Text analysis, pattern detection |
""")
