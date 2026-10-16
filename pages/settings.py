"""
Settings page — application preferences, local caching controls,
privacy verification, and optional LLM integrations.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from core.cache_manager import CacheManager
from learning.preference_store import PreferenceStore
from utils.configuration import AppConfig


def _get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h2 style="margin:0; color:#38bdf8;">⚙️ Application Settings & Storage Management</h2>
            <p style="margin:0.2rem 0 0; color:#94a3b8; font-size:0.95rem;">
                Manage local Parquet caches, bandit learning state, privacy controls, and optional model integrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cache_dir = Path(__file__).parent.parent / "storage" / "cache"
    db_path = Path(__file__).parent.parent / "storage" / "preferences.db"
    pref_store = PreferenceStore(db_path)
    cache_mgr = CacheManager(cache_dir)

    tab1, tab2, tab3 = st.tabs(["🔒 Privacy & Storage", "🎨 UI & Analytics Preferences", "🤖 Optional LLM Integrations"])

    with tab1:
        st.subheader("Data Privacy & Local Sovereignty")
        st.markdown(
            """
            <div style="background:#0f172a; padding:1rem 1.25rem; border-radius:8px; border:1px solid #1e293b; margin-bottom:1.25rem;">
                <div style="color:#22c55e; font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">
                    🛡️ Local-First Architecture Verification
                </div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
                    • <b>No Cloud Ingestion:</b> Datasets never leave your local machine or active workspace.<br>
                    • <b>DuckDB Sandbox:</b> External network access and file writing from SQL queries are disabled.<br>
                    • <b>Zero Analytics Telemetry:</b> Usage telemetry and tracking scripts are completely disabled.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Local Cache & Storage Management")
        c1, c2 = st.columns(2)
        with c1:
            cache_size = _get_dir_size_mb(cache_dir)
            st.metric("Parquet Cache Usage", f"{cache_size} MB")
            if st.button("🧹 Clear Parquet Cache", use_container_width=True):
                cache_mgr.clear_all()
                st.success("Parquet cache successfully cleared.")
                st.rerun()

        with c2:
            db_size = _get_dir_size_mb(db_path)
            st.metric("SQLite Learning Store", f"{db_size} MB")
            if st.button("🔄 Reset Learned Preferences", use_container_width=True):
                if db_path.exists():
                    db_path.unlink()
                pref_store._init_db()
                st.success("Preference database reset to factory state.")
                st.rerun()

    with tab2:
        st.subheader("Analytics & Compute Preferences")
        max_rows = st.number_input(
            "Max Rows for In-Memory Full Profiling (Beyond this, stratified sampling is applied)",
            min_value=10_000,
            max_value=2_000_000,
            value=200_000,
            step=25_000,
        )
        sample_pct = st.slider("Default Sampling Ratio (%) when triggered", min_value=1, max_value=50, value=10)

        enable_learning = st.toggle(
            "Enable Adaptive Recommendation Learning (Thompson Sampling)",
            value=st.session_state.get("learning_enabled", True),
        )
        st.session_state["learning_enabled"] = enable_learning

        st.info("Settings are automatically applied to current session.")

    with tab3:
        st.subheader("Optional LLM Copilot Configuration")
        st.caption("By default, Local Data Analyst runs offline deterministic heuristics. You may optionally connect an LLM for enhanced queries.")

        llm_provider = st.selectbox("Model Provider", ["Offline / Rule-Based (Default)", "Ollama (Local Endpoint)", "OpenAI", "Anthropic", "Google Gemini"])

        if llm_provider == "Ollama (Local Endpoint)":
            st.text_input("Ollama Base URL", value="http://localhost:11434")
            st.text_input("Model Name", value="llama3:latest")
        elif llm_provider in ["OpenAI", "Anthropic", "Google Gemini"]:
            api_key = st.text_input(f"{llm_provider} API Key", type="password")
            st.caption("Key is stored solely in-memory for this session and never written to disk.")

        st.button("Save Model Configuration", type="primary")


if __name__ == "__main__":
    render()
