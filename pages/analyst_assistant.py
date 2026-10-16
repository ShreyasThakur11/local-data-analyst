"""
Analyst Assistant page — conversational analytics copilot.
Translates natural language analytical questions into safe DuckDB queries and charts.
Works fully offline with deterministic query parsing, with optional LLM augmentation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import BASE_LAYOUT, bar_chart, line_chart, scatter_plot
from components.metric_cards import active_dataset
from core.query_engine import _get_conn, register_dataset, run_query
from utils.security import is_read_only_sql


def _parse_nl_query(query: str, df: pd.DataFrame, table_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Deterministic rule-based NL to SQL parser for offline usage.
    Returns: (sql, chart_type, explanation)
    """
    q = query.strip().lower()
    cols = list(df.columns)
    col_map = {c.lower(): c for c in cols}
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]

    # 1. Row count / summary
    if re.search(r"\b(how many (rows|records)|total rows|count records|size of)\b", q):
        sql = f'SELECT COUNT(*) AS total_records FROM "{table_name}"'
        return sql, "metric", "Calculated total row count for the dataset."

    # 2. Check for Top / Highest / Largest N
    top_match = re.search(r"\b(top|highest|largest|biggest|best)\s+(\d+)?\s*(.*)", q)
    if top_match:
        limit = int(top_match.group(2)) if top_match.group(2) else 10
        # identify target column
        found_num = None
        found_cat = None
        for col_lower, orig in col_map.items():
            if col_lower in q:
                if orig in num_cols and found_num is None:
                    found_num = orig
                elif orig in cat_cols and found_cat is None:
                    found_cat = orig
        if not found_num and num_cols:
            found_num = num_cols[0]
        if not found_cat and cat_cols:
            found_cat = cat_cols[0]

        if found_num and found_cat:
            sql = f'SELECT "{found_cat}", SUM("{found_num}") AS total_{found_num} FROM "{table_name}" GROUP BY "{found_cat}" ORDER BY total_{found_num} DESC LIMIT {limit}'
            return sql, "bar", f"Aggregated {found_num} grouped by {found_cat}, ordered highest first (top {limit})."
        elif found_num:
            sql = f'SELECT * FROM "{table_name}" ORDER BY "{found_num}" DESC LIMIT {limit}'
            return sql, "table", f"Retrieved top {limit} records ordered by {found_num} descending."

    # 3. Check for Bottom / Lowest / Smallest N
    bot_match = re.search(r"\b(bottom|lowest|smallest|worst)\s+(\d+)?\s*(.*)", q)
    if bot_match:
        limit = int(bot_match.group(2)) if bot_match.group(2) else 10
        found_num = None
        found_cat = None
        for col_lower, orig in col_map.items():
            if col_lower in q:
                if orig in num_cols and found_num is None:
                    found_num = orig
                elif orig in cat_cols and found_cat is None:
                    found_cat = orig
        if not found_num and num_cols:
            found_num = num_cols[0]
        if not found_cat and cat_cols:
            found_cat = cat_cols[0]

        if found_num and found_cat:
            sql = f'SELECT "{found_cat}", SUM("{found_num}") AS total_{found_num} FROM "{table_name}" GROUP BY "{found_cat}" ORDER BY total_{found_num} ASC LIMIT {limit}'
            return sql, "bar", f"Aggregated {found_num} grouped by {found_cat}, ordered lowest first (bottom {limit})."
        elif found_num:
            sql = f'SELECT * FROM "{table_name}" ORDER BY "{found_num}" ASC LIMIT {limit}'
            return sql, "table", f"Retrieved bottom {limit} records ordered by {found_num} ascending."

    # 4. Average / Mean query
    if "average" in q or "avg" in q or "mean" in q:
        target_num = None
        target_group = None
        for col_lower, orig in col_map.items():
            if col_lower in q:
                if orig in num_cols and target_num is None:
                    target_num = orig
                elif orig in cat_cols and target_group is None:
                    target_group = orig
        if not target_num and num_cols:
            target_num = num_cols[0]

        if target_group and target_num:
            sql = f'SELECT "{target_group}", ROUND(AVG("{target_num}"), 2) AS avg_{target_num} FROM "{table_name}" GROUP BY "{target_group}" ORDER BY avg_{target_num} DESC'
            return sql, "bar", f"Calculated mean {target_num} grouped by {target_group}."
        elif target_num:
            sql = f'SELECT ROUND(AVG("{target_num}"), 2) AS avg_{target_num}, ROUND(MIN("{target_num}"), 2) AS min_val, ROUND(MAX("{target_num}"), 2) AS max_val FROM "{table_name}"'
            return sql, "metric", f"Computed average, minimum, and maximum for {target_num}."

    # 5. Trend / Time-series
    if "trend" in q or "over time" in q or "by date" in q or "monthly" in q:
        date_col = date_cols[0] if date_cols else None
        target_num = num_cols[0] if num_cols else None
        for col_lower, orig in col_map.items():
            if col_lower in q and orig in num_cols:
                target_num = orig
            if col_lower in q and orig in date_cols:
                date_col = orig

        if date_col and target_num:
            sql = f'SELECT "{date_col}", SUM("{target_num}") AS "{target_num}" FROM "{table_name}" GROUP BY "{date_col}" ORDER BY "{date_col}" ASC'
            return sql, "line", f"Time-series trend of {target_num} across {date_col}."

    # 6. Breakdown / Group by
    if "breakdown" in q or "by" in q or "per" in q or "distribution" in q:
        target_num = None
        target_cat = None
        for col_lower, orig in col_map.items():
            if col_lower in q:
                if orig in num_cols and target_num is None:
                    target_num = orig
                elif orig in cat_cols and target_cat is None:
                    target_cat = orig
        if target_cat and target_num:
            sql = f'SELECT "{target_cat}", SUM("{target_num}") AS total_{target_num} FROM "{table_name}" GROUP BY "{target_cat}" ORDER BY total_{target_num} DESC LIMIT 15'
            return sql, "bar", f"Breakdown of {target_num} grouped by {target_cat}."
        elif target_cat:
            sql = f'SELECT "{target_cat}", COUNT(*) AS frequency FROM "{table_name}" GROUP BY "{target_cat}" ORDER BY frequency DESC LIMIT 15'
            return sql, "bar", f"Distribution and count of records across {target_cat}."

    # Default fallback: Sample preview
    sql = f'SELECT * FROM "{table_name}" LIMIT 10'
    return sql, "table", "Displaying the first 10 sample records from the dataset."


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h2 style="margin:0; color:#38bdf8;">🤖 Analyst Assistant</h2>
            <p style="margin:0.2rem 0 0; color:#94a3b8; font-size:0.9rem;">
                Natural language query interface and SQL copilot. Ask questions in plain English or write custom DuckDB SQL.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, profile, rec = active_dataset()
    if df is None:
        return

    table_name = rec.get("name", "active_data").replace(" ", "_").replace("-", "_")
    register_dataset(table_name, df)

    tab1, tab2 = st.tabs(["💬 Natural Language Analyst", "⚡ Direct SQL Console"])

    with tab1:
        st.markdown(
            """
            <div style="background:#1e293b; padding:0.8rem 1rem; border-radius:8px; border:1px solid #334155; margin-bottom:1rem; font-size:0.85rem; color:#cbd5e1;">
                💡 <b>Quick Example Questions you can ask:</b><br>
                • <i>"What are the top 5 records by highest values?"</i><br>
                • <i>"Show average breakdown by category"</i><br>
                • <i>"How many rows are in this dataset?"</i><br>
                • <i>"Show trend over time"</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_query = st.chat_input("Ask a question about your active dataset...")

        # Initialize assistant chat history
        if "assistant_history" not in st.session_state:
            st.session_state["assistant_history"] = []

        # Render past history
        for msg in st.session_state["assistant_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sql" in msg:
                    with st.expander("Generated DuckDB SQL"):
                        st.code(msg["sql"], language="sql")
                if "df" in msg and msg["df"] is not None:
                    st.dataframe(msg["df"], use_container_width=True)

        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate query
            sql, chart_type, explanation = _parse_nl_query(user_query, df, table_name)

            with st.chat_message("assistant"):
                st.markdown(explanation)
                with st.expander("Generated SQL Query", expanded=False):
                    st.code(sql, language="sql")

                res = run_query(sql)
                if not res["ok"]:
                    st.error(f"Execution error: {res['error']}")
                else:
                    res_df = res["df"]
                    if res_df is not None and not res_df.empty:
                        if chart_type == "bar" and len(res_df.columns) >= 2:
                            x_col, y_col = res_df.columns[0], res_df.columns[1]
                            fig = px.bar(res_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == "line" and len(res_df.columns) >= 2:
                            x_col, y_col = res_df.columns[0], res_df.columns[1]
                            fig = px.line(res_df, x=x_col, y=y_col, markers=True, title=f"{y_col} over {x_col}", template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == "metric" and len(res_df.columns) == 1 and len(res_df) == 1:
                            val = res_df.iloc[0, 0]
                            st.metric(label=res_df.columns[0], value=f"{val:,}" if isinstance(val, (int, float)) else str(val))
                        
                        st.dataframe(res_df, use_container_width=True)

                        # Save to history
                        st.session_state["assistant_history"].append({
                            "role": "user",
                            "content": user_query,
                        })
                        st.session_state["assistant_history"].append({
                            "role": "assistant",
                            "content": explanation,
                            "sql": sql,
                            "df": res_df,
                        })

    with tab2:
        st.subheader("DuckDB SQL Studio")
        st.caption(f"Query active table: `{table_name}`. Read-only queries permitted.")

        default_sql = f'SELECT * FROM "{table_name}" LIMIT 15'
        sql_input = st.text_area("SQL Statement", value=default_sql, height=130)

        c1, c2 = st.columns([1, 4])
        with c1:
            run_btn = st.button("🚀 Execute SQL", use_container_width=True)

        if run_btn:
            ok, reason = is_read_only_sql(sql_input)
            if not ok:
                st.error(f"Query validation failed: {reason}")
            else:
                result = run_query(sql_input)
                if not result["ok"]:
                    st.error(f"SQL Error: {result['error']}")
                else:
                    out_df = result["df"]
                    st.success(f"Returned {len(out_df):,} rows and {len(out_df.columns)} columns.")
                    st.dataframe(out_df, use_container_width=True)

                    # Quick CSV download
                    csv_data = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Query Results (CSV)",
                        csv_data,
                        file_name=f"query_result_{table_name}.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    render()
