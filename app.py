# Local Data Analyst
# ==================
# Application entry point. Run with:
#   streamlit run app.py

import os
import sys
from pathlib import Path

# Ensure local packages are importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Load env vars from .env if present
from dotenv import load_dotenv
load_dotenv()

from utils.configuration import AppConfig
from utils.logging_config import setup_logging

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**Local Data Analyst** – Personal analytics copilot.\n\n"
            "All processing is local. No data is sent to external services by default."
        ),
    },
)

setup_logging()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar nav */
.css-1d391kg { background: #0f172a; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-card .kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.kpi-card .kpi-delta {
    font-size: 0.8rem;
    color: #38bdf8;
}

/* Insight chips */
.tag-chip {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.25rem;
    margin-bottom: 0.25rem;
}
.tag-numeric   { background:#1d4ed8; color:#bfdbfe; }
.tag-category  { background:#7c3aed; color:#ddd6fe; }
.tag-date      { background:#0f766e; color:#99f6e4; }
.tag-text      { background:#b45309; color:#fde68a; }
.tag-id        { background:#374151; color:#d1d5db; }
.tag-bool      { background:#166534; color:#bbf7d0; }
.tag-unknown   { background:#6b7280; color:#f3f4f6; }

/* Quality badge */
.quality-good    { color: #22c55e; }
.quality-warn    { color: #f59e0b; }
.quality-bad     { color: #ef4444; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session-state defaults ──────────────────────────────────────────────────────
defaults = {
    "workspace": {},           # {dataset_id: DatasetRecord}
    "active_dataset_id": None,
    "profile": "General Data Analysis",
    "column_roles": {},        # {dataset_id: {col: role}}
    "transformation_logs": {},
    "dashboard_panels": [],
    "learning_enabled": True,
    "assistant_history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = {
    "🏠 Home & Workspace":          "pages/workspace.py",
    "📋 Dataset Overview":           "pages/dataset_overview.py",
    "🔍 Column Explorer":            "pages/column_explorer.py",
    "🩺 Data Quality":               "pages/data_quality.py",
    "🗂️ Column Mapping":             "pages/column_mapping.py",
    "📈 Relationship Analysis":      "pages/relationship_analysis.py",
    "🧪 Advanced Analytics":         "pages/advanced_analysis.py",
    "📊 Dashboard Builder":          "pages/dashboard_builder.py",
    "💼 Consultant Mode":            "pages/consultant_mode.py",
    "🤖 Analyst Assistant":          "pages/analyst_assistant.py",
    "🎓 Learning Center":            "pages/learning_center.py",
    "📄 Report Export":              "pages/report_export.py",
    "⚙️ Settings":                   "pages/settings.py",
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1rem; border-bottom:1px solid #334155; margin-bottom:1rem;">
            <div style="font-size:1.3rem; font-weight:700; color:#38bdf8;">📊 Local Data Analyst</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">Personal Analytics Copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_label = st.selectbox(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    # Active dataset indicator
    active_id = st.session_state.get("active_dataset_id")
    if active_id and active_id in st.session_state["workspace"]:
        rec = st.session_state["workspace"][active_id]
        st.markdown(
            f"""
            <div style="margin-top:1rem; padding:0.6rem 0.8rem; background:#1e293b;
                        border-radius:8px; border:1px solid #334155; font-size:0.75rem;">
                <div style="color:#94a3b8;">Active dataset</div>
                <div style="color:#e2e8f0; font-weight:600; margin-top:0.2rem;">
                    {rec.get('name','Unknown')}
                </div>
                <div style="color:#64748b; margin-top:0.1rem;">
                    {rec.get('shape_str','–')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Profile selector
    profile_options = [
        "General Data Analysis",
        "MBA Statistics & Research",
        "Finance & Financial Modelling",
        "Marketing & Customer Analytics",
        "Strategy & Business Performance",
        "Operations & Process Improvement",
        "Supply Chain & Logistics",
        "Energy & Natural Resources",
        "Manufacturing & Industrial",
        "Sustainability & ESG",
        "Custom Analysis",
    ]
    st.session_state["profile"] = st.selectbox(
        "🎯 Analysis Profile",
        profile_options,
        index=profile_options.index(st.session_state.get("profile", "General Data Analysis")),
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.65rem; color:#475569; line-height:1.5;'>"
        "All data processed locally.<br>"
        "No external uploads by default."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Route to selected page ─────────────────────────────────────────────────────
page_path = Path(__file__).parent / PAGES[page_label]

if page_path.exists():
    with open(page_path, encoding="utf-8") as fh:
        exec(compile(fh.read(), str(page_path), "exec"), {"__name__": "__main__", "__file__": str(page_path)})  # noqa: S102
else:
    st.error(f"Page not found: {page_path}")
    st.info("This page may not yet be implemented. Check the project roadmap.")
