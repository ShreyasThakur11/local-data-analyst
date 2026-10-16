"""
Report Export page — generates executive briefings, quality audit scorecards,
and technical reports in standalone HTML, Markdown, and JSON formats.
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from components.metric_cards import active_dataset
from core.insights_engine import summarise_dataset


def _generate_html_report(title: str, author: str, org: str, df: pd.DataFrame, profile: Any, sections: list[str]) -> str:
    """Generate self-contained, publication-ready HTML report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Statistical summary table HTML
    num_cols = profile.numeric_cols[:10]
    if num_cols:
        stat_df = df[num_cols].describe().round(3).T
        stats_html = stat_df.to_html(classes="styled-table")
    else:
        stats_html = "<p>No numeric columns available.</p>"

    # Data dictionary HTML
    dict_rows = []
    for c in df.columns[:25]:
        dict_rows.append(f"<tr><td><b>{c}</b></td><td>{df[c].dtype}</td><td>{df[c].nunique()}</td><td>{df[c].isna().sum()}</td></tr>")
    dict_html = f"""
    <table class="styled-table">
        <thead><tr><th>Column</th><th>Dtype</th><th>Distinct</th><th>Missing</th></tr></thead>
        <tbody>{''.join(dict_rows)}</tbody>
    </table>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background: #f8fafc;
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }}
    .header {{
        border-bottom: 2px solid #0284c7;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    h1 {{ color: #0f172a; margin: 0 0 10px 0; font-size: 28px; }}
    .meta {{ color: #64748b; font-size: 14px; margin-bottom: 5px; }}
    h2 {{ color: #0369a1; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-top: 30px; }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 20px 0;
    }}
    .kpi-card {{
        background: #f1f5f9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #cbd5e1;
    }}
    .kpi-val {{ font-size: 22px; font-weight: bold; color: #0f172a; margin-top: 5px; }}
    .kpi-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
    .styled-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 13px;
    }}
    .styled-table th, .styled-table td {{
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
        text-align: left;
    }}
    .styled-table th {{ background: #f8fafc; font-weight: 600; color: #334155; }}
    .footer {{
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        font-size: 12px;
        color: #94a3b8;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{title}</h1>
        <div class="meta"><b>Organization:</b> {org} | <b>Author:</b> {author}</div>
        <div class="meta"><b>Generated At:</b> {now_str} (Local Time)</div>
        <div class="meta"><b>Source Dataset:</b> {profile.source_name} ({len(df):,} rows × {len(df.columns)} columns)</div>
    </div>
"""

    if "Executive Summary & KPIs" in sections:
        html += f"""
    <h2>Executive Summary & KPI Scorecard</h2>
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-val">{profile.n_rows:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Columns</div>
            <div class="kpi-val">{profile.n_cols}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Quality Score</div>
            <div class="kpi-val">{profile.quality_score}/100</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Duplicates</div>
            <div class="kpi-val">{profile.duplicate_rows:,}</div>
        </div>
    </div>
    <p>This report presents an analytical evaluation of the <b>{profile.source_name}</b> dataset. Total memory consumption is {profile.memory_mb} MB with a total missingness rate of {profile.total_missing_pct}%.</p>
"""

    if "Data Quality Audit" in sections:
        warns = "".join([f"<li>{w}</li>" for w in profile.warnings]) if profile.warnings else "<li>No critical quality anomalies flagged.</li>"
        html += f"""
    <h2>Data Quality & Integrity Audit</h2>
    <p>Overall quality evaluation yielded <b>{profile.quality_score}/100</b>.</p>
    <ul>{warns}</ul>
"""

    if "Column Schema & Dictionary" in sections:
        html += f"""
    <h2>Data Dictionary & Schema Layout</h2>
    {dict_html}
"""

    if "Descriptive Statistics" in sections:
        html += f"""
    <h2>Descriptive Statistics Summary</h2>
    {stats_html}
"""

    html += f"""
    <div class="footer">
        Generated by Local Data Analyst • Local-First Analytical Copilot • All computations executed locally.
    </div>
</div>
</body>
</html>
"""
    return html


def _generate_markdown_report(title: str, author: str, org: str, df: pd.DataFrame, profile: Any, sections: list[str]) -> str:
    """Generate cleanly formatted Markdown report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"**Organization:** {org}  ",
        f"**Author:** {author}  ",
        f"**Generated:** {now_str}  ",
        f"**Dataset:** `{profile.source_name}` ({profile.n_rows:,} rows × {profile.n_cols} columns)  ",
        "\n---",
    ]

    if "Executive Summary & KPIs" in sections:
        lines.extend([
            "\n## 1. Executive Summary & KPIs",
            f"- **Total Rows:** {profile.n_rows:,}",
            f"- **Total Columns:** {profile.n_cols}",
            f"- **Data Quality Score:** {profile.quality_score}/100",
            f"- **Memory Usage:** {profile.memory_mb} MB",
            f"- **Duplicate Rows:** {profile.duplicate_rows:,}",
            f"- **Total Missing Cells:** {profile.total_missing:,} ({profile.total_missing_pct}%)",
        ])

    if "Data Quality Audit" in sections:
        lines.extend([
            "\n## 2. Data Quality Audit",
            f"Quality Score: **{profile.quality_score}/100**",
        ])
        if profile.warnings:
            lines.append("\n**Flagged Warnings:**")
            for w in profile.warnings:
                lines.append(f"- {w}")
        else:
            lines.append("\n- No severe data hygiene defects detected.")

    if "Column Schema & Dictionary" in sections:
        lines.extend([
            "\n## 3. Data Dictionary",
            "| Column | Type | Distinct Values | Missing Count |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for c in df.columns[:30]:
            lines.append(f"| `{c}` | {df[c].dtype} | {df[c].nunique():,} | {df[c].isna().sum():,} |")

    if "Descriptive Statistics" in sections:
        lines.extend([
            "\n## 4. Key Numeric Statistics",
        ])
        num_cols = profile.numeric_cols[:10]
        if num_cols:
            stat_df = df[num_cols].describe().round(3).T
            lines.append(stat_df.to_markdown())
        else:
            lines.append("No continuous numeric variables present.")

    lines.extend([
        "\n---",
        "*Generated with Local Data Analyst • Microsoft Enterprise Architecture Standards*",
    ])
    return "\n".join(lines)


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h2 style="margin:0; color:#38bdf8;">📄 Executive & Technical Report Export</h2>
            <p style="margin:0.2rem 0 0; color:#94a3b8; font-size:0.95rem;">
                Compile comprehensive analytical findings, quality audit scores, and statistical profiles into publication-ready reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, profile, rec = active_dataset()
    if df is None:
        return

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Report Configuration")
        title = st.text_input("Report Title", value=f"Analytical Audit: {rec.get('name', 'Dataset')}")
        author = st.text_input("Author Name", value="Principal Data Analyst")
        org = st.text_input("Organization", value="Enterprise Analytics")

        st.markdown("**Sections to Include**")
        sec_kpi = st.checkbox("Executive Summary & KPIs", value=True)
        sec_audit = st.checkbox("Data Quality Audit", value=True)
        sec_dict = st.checkbox("Column Schema & Dictionary", value=True)
        sec_stats = st.checkbox("Descriptive Statistics", value=True)

        chosen_sections = []
        if sec_kpi: chosen_sections.append("Executive Summary & KPIs")
        if sec_audit: chosen_sections.append("Data Quality Audit")
        if sec_dict: chosen_sections.append("Column Schema & Dictionary")
        if sec_stats: chosen_sections.append("Descriptive Statistics")

        fmt = st.radio("Export Format", ["Standalone HTML", "Markdown (.md)", "JSON Payload"], horizontal=True)

    with c2:
        st.subheader("Report Preview & Download")

        if fmt == "Standalone HTML":
            html_out = _generate_html_report(title, author, org, df, profile, chosen_sections)
            st.download_button(
                "📥 Download Standalone HTML Report",
                data=html_out,
                file_name=f"report_{rec.get('name', 'dataset')}.html",
                mime="text/html",
                use_container_width=True,
            )
            with st.expander("Interactive HTML Preview", expanded=True):
                st.components.v1.html(html_out, height=500, scrolling=True)

        elif fmt == "Markdown (.md)":
            md_out = _generate_markdown_report(title, author, org, df, profile, chosen_sections)
            st.download_button(
                "📥 Download Markdown Report",
                data=md_out,
                file_name=f"report_{rec.get('name', 'dataset')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            with st.expander("Markdown Source Preview", expanded=True):
                st.code(md_out, language="markdown")

        elif fmt == "JSON Payload":
            payload = {
                "title": title,
                "author": author,
                "organization": org,
                "generated_at": datetime.now().isoformat(),
                "dataset": {
                    "name": profile.source_name,
                    "rows": profile.n_rows,
                    "cols": profile.n_cols,
                    "memory_mb": profile.memory_mb,
                    "quality_score": profile.quality_score,
                    "warnings": profile.warnings,
                },
                "columns": [
                    {"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique())}
                    for c in df.columns
                ],
            }
            json_str = json.dumps(payload, indent=2)
            st.download_button(
                "📥 Download JSON Analytical Payload",
                data=json_str,
                file_name=f"report_{rec.get('name', 'dataset')}.json",
                mime="application/json",
                use_container_width=True,
            )
            with st.expander("JSON Preview", expanded=True):
                st.json(payload)


if __name__ == "__main__":
    render()
