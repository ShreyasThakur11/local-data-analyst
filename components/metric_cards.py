"""
Reusable Streamlit UI components: KPI cards, metric displays, feedback controls.
"""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
import streamlit as st


def kpi_card(label: str, value: str, delta: Optional[str] = None, icon: str = "") -> None:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def type_badge(inferred_type: str) -> str:
    """Return an HTML tag chip for a given type string."""
    t = inferred_type.lower()
    css_class = (
        "tag-numeric" if "numeric" in t or "currency" in t or "percent" in t
        else "tag-category" if "categorical" in t or "ordinal" in t
        else "tag-date" if "date" in t
        else "tag-text" if "text" in t
        else "tag-id" if "identifier" in t
        else "tag-bool" if "boolean" in t
        else "tag-unknown"
    )
    return f'<span class="tag-chip {css_class}">{inferred_type}</span>'


def quality_badge(pct_missing: float) -> str:
    if pct_missing == 0:
        return '<span class="quality-good">✓ Complete</span>'
    elif pct_missing < 10:
        return f'<span class="quality-warn">⚠ {pct_missing:.1f}% missing</span>'
    else:
        return f'<span class="quality-bad">✗ {pct_missing:.1f}% missing</span>'


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_data_dictionary(profiles: dict, overrides: Optional[dict] = None) -> None:
    """Render the column data dictionary as a styled table."""
    rows = []
    for col_name, cp in profiles.items():
        role_override = (overrides or {}).get(col_name)
        rows.append({
            "Column": col_name,
            "Original Type": cp.original_type,
            "Inferred Type": cp.inferred_type.value,
            "Semantic Role": (role_override or cp.semantic_role.value),
            "Non-Missing": f"{cp.non_missing:,}",
            "Missing %": f"{cp.missing_pct:.1f}%",
            "Unique Values": f"{cp.unique_count:,}",
            "Examples": ", ".join(str(v) for v in cp.example_values[:3]),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def feedback_controls(rec_id: str, context: Optional[dict] = None) -> Optional[str]:
    """
    Render feedback buttons for a recommendation.
    Returns the selected rating string, or None.
    """
    cols = st.columns(5)
    rating = None

    labels = [
        ("👍 Useful", "useful"),
        ("👎 Not useful", "not_useful"),
        ("⚠️ Incorrect", "incorrect"),
        ("✏️ Modified", "modified"),
        ("⭐ Save", "saved"),
    ]

    for i, (label, value) in enumerate(labels):
        btn_key = f"feedback_{rec_id}_{value}"
        if cols[i].button(label, key=btn_key, use_container_width=True):
            rating = value

    return rating


def recommendation_card(rec, on_run=None) -> None:
    """Render a single recommendation card."""
    with st.expander(f"💡 {rec.title}", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Why:** {rec.why}")
            st.markdown(f"**Method:** {rec.method}")
            st.markdown(f"**Output:** {rec.output}")
            st.markdown(f"**Columns:** `{'`, `'.join(rec.columns)}`")
            st.caption(f"⚠️ {rec.limitations}")
        with col2:
            priority_pct = int(rec.priority * 100)
            st.metric("Priority", f"{priority_pct}%")

        if on_run:
            if st.button(f"▶ Run Analysis", key=f"run_{rec.id}"):
                on_run(rec)

        return feedback_controls(rec.id)


def active_dataset_guard() -> Optional[Any]:
    """
    Check that an active dataset is loaded.
    Returns the dataset record dict, or None if not loaded.
    Shows a helpful message if not.
    """
    active_id = st.session_state.get("active_dataset_id")
    workspace = st.session_state.get("workspace", {})

    if not active_id or active_id not in workspace:
        st.info(
            "📂 No dataset loaded. Go to **🏠 Home & Workspace** to upload or select a file.",
            icon="ℹ️",
        )
        return None

    return workspace[active_id]


def get_active_df() -> Optional[pd.DataFrame]:
    """Return the active DataFrame or None."""
    rec = active_dataset_guard()
    if rec is None:
        return None
    return rec.get("df")
