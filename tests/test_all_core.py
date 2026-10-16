"""
Comprehensive unit tests for Local Data Analyst core modules.
Tests cover:
- Type inference & semantic roles
- Ingestion & caching
- Profiling & Quality scoring
- Statistics engine (ANOVA, correlation matrix, grouped comparison)
- Data quality audits & safe non-destructive transformations
- Query engine (DuckDB read-only SQL)
- Insights engine
- Preference store (SQLite & Thompson sampling)
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.cache_manager import CacheManager
from core.data_quality import (
    TransformationLog,
    TransformationRecord,
    audit_data_quality,
    preview_drop_duplicates,
    preview_fill_missing,
)
from core.insights_engine import explain_correlation, explain_group_comparison, summarise_dataset
from core.profiling import profile_dataframe
from core.query_engine import aggregate_query, register_dataset, run_query
from core.statistics_engine import (
    correlation_matrix,
    grouped_comparison,
    pearson_correlation,
    spearman_correlation,
)
from core.type_inference import InferredType, infer_all_columns
from learning.preference_store import PreferenceStore
from utils.security import compute_dataframe_hash, compute_file_hash, is_read_only_sql
from utils.validation import is_safe_sql_identifier, sanitise_column_name, validate_file_path


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = np.random.choice(["Electronics", "Furniture", "Supplies"], size=n)
    values = np.random.normal(loc=250.0, scale=40.0, size=n)
    quantities = np.random.randint(1, 10, size=n)

    df = pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(n)],
        "order_date": dates,
        "category": categories,
        "revenue": values,
        "quantity": quantities,
        "is_discounted": np.random.choice([True, False], size=n),
    })
    # Add a few NaNs to test missing handling
    df.loc[5:8, "revenue"] = np.nan
    return df


def test_type_inference(sample_df):
    inferred = infer_all_columns(sample_df)
    assert inferred["order_id"].inferred_type in (InferredType.IDENTIFIER, InferredType.TEXT)
    assert inferred["order_date"].inferred_type == InferredType.DATE
    assert inferred["category"].inferred_type == InferredType.CATEGORICAL
    assert inferred["revenue"].inferred_type in (InferredType.NUMERIC_CONTINUOUS, InferredType.NUMERIC_DISCRETE)
    assert inferred["quantity"].inferred_type in (InferredType.NUMERIC_CONTINUOUS, InferredType.NUMERIC_DISCRETE)
    assert inferred["is_discounted"].inferred_type == InferredType.BOOLEAN


def test_profiling(sample_df):
    profile = profile_dataframe(sample_df, source_name="TestOrders")
    assert profile.n_rows == 100
    assert profile.n_cols == 6
    assert 0 <= profile.quality_score <= 100
    assert "revenue" in profile.numeric_cols
    assert profile.total_missing > 0


def test_statistics_engine(sample_df):
    corr = correlation_matrix(sample_df[["revenue", "quantity"]], method="pearson")
    assert "matrix" in corr
    assert len(corr["pairs"]) > 0

    p_corr = pearson_correlation(sample_df["revenue"].dropna(), sample_df["quantity"][: len(sample_df["revenue"].dropna())])
    assert "r" in p_corr
    assert "p_value" in p_corr

    s_corr = spearman_correlation(sample_df["revenue"].dropna(), sample_df["quantity"][: len(sample_df["revenue"].dropna())])
    assert "rho" in s_corr

    grouped = grouped_comparison(sample_df, "category", "revenue")
    assert "stats" in grouped
    assert "anova" in grouped


def test_data_quality_audit(sample_df):
    issues = audit_data_quality(sample_df)
    assert isinstance(issues, list)

    clean_df, n_filled = preview_fill_missing(sample_df, "revenue", method="median")
    assert n_filled == 4
    assert clean_df["revenue"].isna().sum() == 0

    log = TransformationLog()
    log.add(TransformationRecord(
        operation="fill_missing",
        columns=["revenue"],
        parameters={"method": "median"},
        rows_affected=n_filled,
    ))
    log_df = log.to_dataframe()
    assert len(log_df) == 1


def test_query_engine(sample_df):
    register_dataset("test_orders", sample_df)
    res = run_query('SELECT COUNT(*) AS total FROM "test_orders"')
    assert res["ok"] is True
    assert res["df"].iloc[0]["total"] == 100

    agg = aggregate_query("test_orders", "revenue", group_col="category", agg_func="SUM")
    assert agg["ok"] is True
    assert len(agg["df"]) > 0

    blocked = run_query("DROP TABLE test_orders;")
    assert blocked["ok"] is False


def test_security_utilities():
    ok, _ = is_read_only_sql("SELECT * FROM table")
    assert ok is True

    bad, _ = is_read_only_sql("DROP TABLE customers;")
    assert bad is False

    assert is_safe_sql_identifier("valid_column_name") is True
    assert is_safe_sql_identifier("invalid column!") is False
    assert sanitise_column_name(" Gross Sales ($) ") == "gross_sales_"


def test_preference_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "prefs.db"
        store = PreferenceStore(db_path)
        store.set_preference("theme", "dark")
        assert store.get_preference("theme") == "dark"

        store.record_feedback("rec_correlation", "useful")
        scores = store.get_bandit_scores()
        assert "rec_correlation" in scores
        summary = store.get_learning_summary()
        assert summary["total_feedback"] == 1


def test_cache_manager(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(Path(tmpdir))
        cache.cache_dataframe("key1", sample_df)
        cached_df = cache.load_dataframe("key1")
        assert cached_df is not None
        assert len(cached_df) == len(sample_df)
        cache.clear_all()
        assert cache.load_dataframe("key1") is None


def test_insights_engine(sample_df):
    profile = profile_dataframe(sample_df, "sample")
    text = summarise_dataset(profile)
    assert "Dataset:" in text
    assert "Overall data quality score:" in text

    corr_exp = explain_correlation(0.75, "revenue", "quantity", 100, 0.001)
    assert "positively" in corr_exp
    assert "statistically significant" in corr_exp
