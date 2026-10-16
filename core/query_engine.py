"""
DuckDB-powered query engine for analytical SQL workloads.
All queries are restricted to read-only SELECT statements.
Approved datasets are registered as named views.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from utils.security import is_read_only_sql

logger = logging.getLogger(__name__)

# Per-session connection (in-memory, no persistence)
_conn: Optional[duckdb.DuckDBPyConnection] = None


def _get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(database=":memory:")
        # Restrict filesystem access
        _conn.execute("SET enable_external_access = FALSE;")
    return _conn


def register_dataset(name: str, df: pd.DataFrame) -> None:
    """Register a DataFrame as a read-only DuckDB view."""
    safe_name = name.replace(" ", "_").replace("-", "_")
    conn = _get_conn()
    conn.register(safe_name, df)
    logger.info("Registered DuckDB view: %s (%d rows)", safe_name, len(df))


def unregister_dataset(name: str) -> None:
    safe_name = name.replace(" ", "_").replace("-", "_")
    try:
        _get_conn().execute(f"DROP VIEW IF EXISTS {safe_name}")
    except Exception:
        pass


def run_query(sql: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Execute a read-only SQL query.
    Returns {"ok": bool, "df": DataFrame, "error": str}.
    """
    ok, reason = is_read_only_sql(sql)
    if not ok:
        return {"ok": False, "error": f"Query blocked: {reason}", "df": None}

    try:
        conn = _get_conn()
        result = conn.execute(sql).df()
        return {"ok": True, "df": result, "error": None}
    except Exception as exc:
        logger.warning("Query error: %s", exc)
        return {"ok": False, "error": str(exc), "df": None}


def list_views() -> List[str]:
    """Return names of all registered views."""
    try:
        result = _get_conn().execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).df()
        return list(result["table_name"])
    except Exception:
        return []


def aggregate_query(
    table: str,
    measure_col: str,
    group_col: Optional[str] = None,
    agg_func: str = "SUM",
    filters: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """
    Build and execute a safe aggregation query.
    agg_func: SUM, AVG, MEDIAN, COUNT, MIN, MAX
    """
    allowed_aggs = {"SUM", "AVG", "MEAN", "MEDIAN", "COUNT", "MIN", "MAX", "COUNT_DISTINCT"}
    agg = agg_func.upper()
    if agg == "MEAN":
        agg = "AVG"
    if agg == "COUNT_DISTINCT":
        agg_expr = f'COUNT(DISTINCT "{measure_col}")'
    elif agg in allowed_aggs:
        agg_expr = f'{agg}("{measure_col}")'
    else:
        return {"ok": False, "error": f"Unsupported aggregation: {agg_func}"}

    safe_table = table.replace(" ", "_").replace("-", "_")

    if group_col:
        sql = f'SELECT "{group_col}", {agg_expr} AS value FROM {safe_table}'
    else:
        sql = f"SELECT {agg_expr} AS value FROM {safe_table}"

    if filters:
        # Basic guard — only allow WHERE clauses without subqueries
        if "select" in filters.lower() or ";" in filters:
            return {"ok": False, "error": "Invalid filter expression."}
        sql += f" WHERE {filters}"

    if group_col:
        sql += f' GROUP BY "{group_col}" ORDER BY value DESC LIMIT {int(limit)}'

    return run_query(sql)
