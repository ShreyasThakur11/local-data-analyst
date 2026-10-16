"""
Home & Workspace page.
Handles dataset upload, local path selection, sheet selection, and preview.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.metric_cards import kpi_card, section_header
from core.ingestion import (
    get_excel_sheet_names,
    load_csv,
    load_excel_from_bytes,
    load_parquet,
)
from core.profiling import profile_dataframe
from core.query_engine import register_dataset
from core.type_inference import infer_all_columns
from utils.validation import validate_file_path

st.title("🏠 Home & Workspace")
st.markdown("Load one or more datasets to begin analysis. All processing is local — no data is uploaded to external services.")

# ── Security notice ───────────────────────────────────────────────────────────
with st.expander("⚠️ Data Privacy & Security", expanded=False):
    st.markdown("""
    - All data processing occurs locally on your machine.
    - No data is transmitted to external servers unless you explicitly configure Ollama or another service.
    - Do not load files containing confidential client data unless your employer's data-handling policies permit use on this device and software.
    - Cached data is stored in `storage/cache/`. You can clear it from **Settings**.
    """)

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_upload, tab_local, tab_samples, tab_workspace = st.tabs([
    "📤 Upload File", "📁 Local File Path", "🗂️ Sample Data", "📋 Loaded Datasets"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("Upload a file from your computer. Maximum 200 MB for browser upload.")
    uploaded = st.file_uploader(
        "Choose a file",
        type=["xlsx", "xlsm", "xls", "csv", "tsv", "parquet"],
        key="workspace_uploader",
    )

    if uploaded:
        data = uploaded.read()
        fname = uploaded.name
        suffix = Path(fname).suffix.lower()
        st.info(f"📄 **{fname}** — {len(data)/1024/1024:.2f} MB")

        # Sheet selection for Excel
        sheet_name = None
        header_row = 0

        if suffix in {".xlsx", ".xlsm", ".xls"}:
            with st.spinner("Inspecting workbook…"):
                sheets, warns = get_excel_sheet_names(data, fname)
            if warns:
                for w in warns:
                    st.warning(w)
            if not sheets:
                st.error("Could not read sheet names.")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                sheet_name = st.selectbox("Select worksheet", sheets, key="upload_sheet")
            with col2:
                header_row = st.number_input("Header row (0 = first row)", min_value=0, max_value=20, value=0, key="upload_header")

        col_sep = None
        if suffix in {".csv", ".tsv"}:
            col_sep = st.selectbox("Column separator", [",", ";", "\t", "|", " "], index=0, key="upload_sep")
            if col_sep == " ":
                col_sep = r"\s+"

        if st.button("🔍 Load Dataset", key="btn_load_upload", type="primary"):
            with st.spinner("Loading data…"):
                if suffix in {".xlsx", ".xlsm", ".xls"}:
                    result = load_excel_from_bytes(data, fname, sheet_name=sheet_name, header_row=header_row)
                elif suffix in {".csv", ".tsv"}:
                    result = load_csv(data=data, filename=fname, sep=col_sep)
                elif suffix == ".parquet":
                    result = load_parquet(data=data, filename=fname)
                else:
                    st.error(f"Unsupported format: {suffix}")
                    st.stop()

            if not result.ok:
                for e in result.errors:
                    st.error(e)
                st.stop()

            for w in result.warnings:
                st.warning(w)

            _register_dataset(result, sheet_name=sheet_name)
            st.success(f"✅ Loaded **{fname}** — {result.df.shape[0]:,} rows × {result.df.shape[1]} columns")
            st.dataframe(result.df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Local path
# ─────────────────────────────────────────────────────────────────────────────
with tab_local:
    st.markdown("""
    Specify an absolute path to a file on this machine.
    Useful for large files that would be slow to upload via browser.
    """)

    local_path = st.text_input("Absolute file path", placeholder="/Users/name/data/report.xlsx", key="local_path_input")
    header_row_local = st.number_input("Header row (0 = first row)", min_value=0, max_value=20, value=0, key="local_header")

    if st.button("🔍 Load Local File", key="btn_load_local"):
        if not local_path:
            st.error("Please enter a file path.")
            st.stop()

        ok, msg = validate_file_path(local_path)
        if not ok:
            st.error(msg)
            st.stop()
        if msg != "OK":
            st.warning(msg)

        p = Path(local_path)
        suffix = p.suffix.lower()
        size_mb = p.stat().st_size / 1024 / 1024

        st.info(f"📄 **{p.name}** — {size_mb:.1f} MB")

        if size_mb > 100:
            st.warning(
                f"This file is {size_mb:.0f} MB. Loading may take a moment. "
                "Consider converting to Parquet for repeated analysis."
            )

        with st.spinner("Loading data…"):
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                from core.ingestion import inspect_workbook, load_excel_sheet
                insp = inspect_workbook(p)
                if insp.errors:
                    for e in insp.errors:
                        st.error(e)
                    st.stop()
                if insp.warnings:
                    for w in insp.warnings:
                        st.warning(w)

                sheet_local = st.selectbox("Select worksheet", insp.sheet_names, key="local_sheet")
                result = load_excel_sheet(p, sheet_name=sheet_local, header_row=header_row_local)
            elif suffix in {".csv", ".tsv"}:
                result = load_csv(path=p, filename=p.name)
            elif suffix == ".parquet":
                result = load_parquet(path=p, filename=p.name)
            else:
                st.error(f"Unsupported format: {suffix}")
                st.stop()

        if not result.ok:
            for e in result.errors:
                st.error(e)
            st.stop()

        for w in result.warnings:
            st.warning(w)

        _register_dataset(result)
        st.success(f"✅ Loaded **{p.name}** — {result.df.shape[0]:,} rows × {result.df.shape[1]} columns")
        st.dataframe(result.df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Sample Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_samples:
    st.markdown("Load one of the built-in synthetic datasets for exploration.")
    sample_dir = Path(__file__).parent.parent / "sample_data"
    samples = {
        "📊 Synthetic Sales (500 rows)": sample_dir / "synthetic_sales.csv",
        "⚙️ Synthetic Operations (400 rows)": sample_dir / "synthetic_operations.csv",
        "⚡ Synthetic Energy (300 rows)": sample_dir / "synthetic_energy.csv",
    }

    sample_choice = st.selectbox("Choose sample dataset", list(samples.keys()), key="sample_choice")
    if st.button("📥 Load Sample", key="btn_load_sample", type="primary"):
        path = samples[sample_choice]
        if not path.exists():
            st.error(f"Sample file not found: {path}. Run `python sample_data/generate_samples.py` first.")
            st.stop()

        result = load_csv(path=path, filename=path.name)
        if not result.ok:
            for e in result.errors:
                st.error(e)
            st.stop()

        _register_dataset(result)
        st.success(f"✅ Loaded **{path.name}** — {result.df.shape[0]:,} rows × {result.df.shape[1]} columns")
        st.dataframe(result.df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Workspace overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_workspace:
    workspace = st.session_state.get("workspace", {})
    if not workspace:
        st.info("No datasets loaded yet. Use one of the tabs above to load a dataset.")
    else:
        section_header(f"Loaded Datasets ({len(workspace)})")
        for ds_id, rec in workspace.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            is_active = ds_id == st.session_state.get("active_dataset_id")
            with col1:
                badge = "🟢" if is_active else "⚪"
                st.markdown(f"{badge} **{rec['name']}** — {rec.get('sheet_name','')}")
                st.caption(rec.get("shape_str", ""))
            with col2:
                st.caption(f"Quality: {rec.get('quality_score', '?')}/100")
            with col3:
                if not is_active:
                    if st.button("Set Active", key=f"active_{ds_id}"):
                        st.session_state["active_dataset_id"] = ds_id
                        st.rerun()
                else:
                    st.markdown("✅ Active")

            if st.button("Remove", key=f"remove_{ds_id}"):
                del st.session_state["workspace"][ds_id]
                if st.session_state.get("active_dataset_id") == ds_id:
                    remaining = list(st.session_state["workspace"].keys())
                    st.session_state["active_dataset_id"] = remaining[0] if remaining else None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _register_dataset(result, sheet_name=None) -> None:
    """Register a loaded dataset in session state and DuckDB."""
    df = result.df
    ds_id = f"{result.source_name}_{sheet_name or ''}".replace(" ", "_")

    with st.spinner("Profiling dataset…"):
        profile = profile_dataframe(df, source_name=result.source_name)

    # Register for DuckDB queries
    try:
        register_dataset(ds_id, df)
    except Exception:
        pass

    st.session_state["workspace"][ds_id] = {
        "id": ds_id,
        "name": result.source_name,
        "sheet_name": sheet_name,
        "df": df,
        "profile": profile,
        "shape_str": f"{df.shape[0]:,} rows × {df.shape[1]} cols",
        "quality_score": profile.quality_score,
        "column_roles": {},
        "transformation_log": [],
    }
    st.session_state["active_dataset_id"] = ds_id
