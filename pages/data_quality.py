"""Data Quality page — issues, transformation preview, and logging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.charts import missing_heatmap, quality_score_gauge
from components.metric_cards import active_dataset_guard, section_header
from core.data_quality import (
    TransformationLog, TransformationRecord, audit_data_quality,
    preview_convert_type, preview_drop_duplicates,
    preview_drop_rows_with_missing, preview_fill_missing,
    preview_standardise_categories,
)

st.title("🩺 Data Quality")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df_original = rec["df"]

# Ensure transformation log exists
if "transformation_log" not in rec:
    rec["transformation_log"] = []

# Working copy: apply all approved transformations sequentially
df_working = df_original.copy()
for t in rec["transformation_log"]:
    try:
        df_working = _apply_logged_transform(df_working, t)
    except Exception:
        pass

# ── Quality audit ─────────────────────────────────────────────────────────────
section_header("Quality Audit")

issues = audit_data_quality(df_working)
profile = rec["profile"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Quality Score", f"{profile.quality_score}/100")
high = [i for i in issues if i.severity == "high"]
med  = [i for i in issues if i.severity == "medium"]
low  = [i for i in issues if i.severity == "low"]
c2.metric("🔴 High severity", len(high))
c3.metric("🟡 Medium severity", len(med))
c4.metric("🟢 Low severity", len(low))

st.plotly_chart(quality_score_gauge(profile.quality_score), use_container_width=True)
st.plotly_chart(missing_heatmap(df_working), use_container_width=True)

if not issues:
    st.success("🎉 No significant data quality issues detected!")
else:
    for issue in issues:
        sev_icon = "🔴" if issue.severity == "high" else ("🟡" if issue.severity == "medium" else "🟢")
        with st.expander(f"{sev_icon} {issue.issue_type.replace('_',' ').title()} — {issue.description}", expanded=issue.severity == "high"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Affected rows:** {issue.affected_rows:,} ({issue.pct_affected}%)")
            col2.markdown(f"**Columns:** `{'`, `'.join(issue.columns[:5])}`")
            st.markdown(f"**Recommendation:** {issue.recommendation}")

st.markdown("---")

# ── Transformation panel ───────────────────────────────────────────────────────
section_header("Non-Destructive Transformations")
st.caption("All changes are previewed before application. The original dataset is preserved.")

transform_type = st.selectbox(
    "Choose transformation",
    [
        "None",
        "Remove duplicate rows",
        "Drop rows with missing values",
        "Fill missing values",
        "Convert column type",
        "Standardise category labels",
        "Filter rows",
        "Rename column",
    ],
    key="transform_select",
)

preview_df = None
transform_record = None

if transform_type == "Remove duplicate rows":
    preview_df, n_dropped = preview_drop_duplicates(df_working)
    st.info(f"This will remove **{n_dropped:,}** duplicate rows. Shape after: {preview_df.shape}")
    transform_record = TransformationRecord(
        operation="drop_duplicates",
        columns=list(df_working.columns),
        parameters={},
        rows_affected=n_dropped,
        description=f"Removed {n_dropped} duplicate rows.",
    )

elif transform_type == "Drop rows with missing values":
    cols_to_check = st.multiselect("Columns to check (empty = all)", list(df_working.columns), key="drop_miss_cols")
    thresh = st.slider("Missing fraction threshold (≥ this fraction = drop)", 0.1, 1.0, 1.0, step=0.1, key="drop_miss_thresh")
    preview_df, n_dropped = preview_drop_rows_with_missing(df_working, columns=cols_to_check or None, threshold=thresh)
    st.info(f"This will remove **{n_dropped:,}** rows. Shape after: {preview_df.shape}")
    transform_record = TransformationRecord(
        operation="drop_rows_missing",
        columns=cols_to_check or list(df_working.columns),
        parameters={"threshold": thresh},
        rows_affected=n_dropped,
        description=f"Dropped {n_dropped} rows with missingness ≥ {thresh:.0%}.",
    )

elif transform_type == "Fill missing values":
    fill_col = st.selectbox("Column", list(df_working.columns), key="fill_col")
    fill_method = st.selectbox("Fill method", ["median", "mean", "mode", "zero"], key="fill_method")
    preview_df, n_filled = preview_fill_missing(df_working, fill_col, fill_method)
    st.info(f"Will fill **{n_filled}** missing values in `{fill_col}` with {fill_method}.")
    transform_record = TransformationRecord(
        operation="fill_missing",
        columns=[fill_col],
        parameters={"method": fill_method},
        rows_affected=n_filled,
        description=f"Filled {n_filled} missing values in '{fill_col}' with {fill_method}.",
    )

elif transform_type == "Convert column type":
    conv_col = st.selectbox("Column", list(df_working.columns), key="conv_col")
    target_type = st.selectbox("Target type", ["numeric", "datetime", "string", "category"], key="conv_type")
    preview_df, n_conv, conv_warns = preview_convert_type(df_working, conv_col, target_type)
    for w in conv_warns:
        st.warning(w)
    st.info(f"Will convert `{conv_col}` to {target_type}.")
    transform_record = TransformationRecord(
        operation="convert_type",
        columns=[conv_col],
        parameters={"target_type": target_type},
        rows_affected=n_conv,
        description=f"Converted '{conv_col}' to {target_type}.",
    )

elif transform_type == "Standardise category labels":
    std_col = st.selectbox("Column", list(df_working.select_dtypes("object").columns), key="std_col")
    case = st.selectbox("Case normalisation", ["title", "lower", "upper"], key="std_case")
    preview_df = preview_standardise_categories(df_working, std_col, case)
    st.info(f"Will standardise `{std_col}` labels to {case} case.")
    transform_record = TransformationRecord(
        operation="standardise_categories",
        columns=[std_col],
        parameters={"case": case},
        rows_affected=int(df_working[std_col].notna().sum()),
        description=f"Standardised '{std_col}' to {case} case.",
    )

elif transform_type == "Filter rows":
    filter_col = st.selectbox("Filter column", list(df_working.columns), key="filt_col")
    filter_op = st.selectbox("Condition", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="filt_op")
    filter_val = st.text_input("Value", key="filt_val")
    if filter_val:
        try:
            if filter_op == "contains":
                mask = df_working[filter_col].astype(str).str.contains(filter_val, na=False)
            else:
                # Try numeric comparison first
                try:
                    v = float(filter_val)
                    mask = df_working[filter_col].apply(lambda x: eval(f"{x} {filter_op} {v}") if pd.notna(x) else False)
                except Exception:
                    mask = df_working[filter_col].astype(str).apply(lambda x: eval(f'"{x}" {filter_op} "{filter_val}"'))
            preview_df = df_working[mask]
            st.info(f"Filter keeps **{len(preview_df):,}** of {len(df_working):,} rows.")
        except Exception as e:
            st.error(f"Filter error: {e}")

elif transform_type == "Rename column":
    old_name = st.selectbox("Column to rename", list(df_working.columns), key="rename_old")
    new_name = st.text_input("New name", value=old_name, key="rename_new")
    if new_name and new_name != old_name:
        preview_df = df_working.rename(columns={old_name: new_name})
        st.info(f"Will rename `{old_name}` → `{new_name}`.")
        transform_record = TransformationRecord(
            operation="rename_column",
            columns=[old_name],
            parameters={"new_name": new_name},
            rows_affected=len(df_working),
            description=f"Renamed '{old_name}' to '{new_name}'.",
        )

# ── Preview ───────────────────────────────────────────────────────────────────
if preview_df is not None:
    st.markdown("**Preview (first 10 rows):**")
    st.dataframe(preview_df.head(10), use_container_width=True)

    col_apply, col_cancel = st.columns(2)
    with col_apply:
        if st.button("✅ Apply Transformation", type="primary", key="apply_transform"):
            rec["df"] = preview_df
            if transform_record:
                rec["transformation_log"].append(vars(transform_record))
            st.session_state["workspace"][st.session_state["active_dataset_id"]] = rec
            st.success("Transformation applied. Dataset updated.")
            st.rerun()

# ── Transformation log ────────────────────────────────────────────────────────
st.markdown("---")
section_header("Transformation Log")
log = rec.get("transformation_log", [])
if not log:
    st.info("No transformations have been applied yet.")
else:
    log_df = pd.DataFrame(log)[["timestamp", "operation", "columns", "rows_affected", "description"]]
    log_df.columns = ["Timestamp", "Operation", "Columns", "Rows Affected", "Description"]
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    if st.button("↩️ Undo Last Transformation", key="undo_transform"):
        if rec["transformation_log"]:
            rec["transformation_log"].pop()
            # Re-apply all remaining transformations from original
            df_w = df_original.copy()
            for t in rec["transformation_log"]:
                try:
                    df_w = _apply_logged_transform(df_w, t)
                except Exception:
                    pass
            rec["df"] = df_w
            st.session_state["workspace"][st.session_state["active_dataset_id"]] = rec
            st.success("Last transformation undone.")
            st.rerun()


def _apply_logged_transform(df: pd.DataFrame, t: dict) -> pd.DataFrame:
    """Re-apply a logged transformation to a dataframe."""
    op = t.get("operation")
    params = t.get("parameters", {})
    cols = t.get("columns", [])

    if op == "drop_duplicates":
        return df.drop_duplicates()
    elif op == "drop_rows_missing":
        thresh = params.get("threshold", 1.0)
        mask = df[cols].isna().mean(axis=1) >= thresh
        return df[~mask]
    elif op == "fill_missing" and cols:
        result = df.copy()
        col = cols[0]
        method = params.get("method", "median")
        if method == "mean":
            fill_val = result[col].mean()
        elif method == "median":
            fill_val = result[col].median()
        elif method == "mode":
            modes = result[col].mode()
            fill_val = modes.iloc[0] if not modes.empty else None
        else:
            fill_val = 0
        if fill_val is not None:
            result[col] = result[col].fillna(fill_val)
        return result
    elif op == "convert_type" and cols:
        result = df.copy()
        target = params.get("target_type", "string")
        if target == "numeric":
            result[cols[0]] = pd.to_numeric(result[cols[0]], errors="coerce")
        elif target == "datetime":
            result[cols[0]] = pd.to_datetime(result[cols[0]], errors="coerce")
        elif target == "string":
            result[cols[0]] = result[cols[0]].astype(str)
        return result
    elif op == "rename_column" and cols:
        new_name = params.get("new_name")
        return df.rename(columns={cols[0]: new_name}) if new_name else df
    return df
