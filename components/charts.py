"""
Plotly chart factory.
All charts are returned as Plotly Figure objects.
Titles, labels, and units must always be specified.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design constants ──────────────────────────────────────────────────────────

TEMPLATE = "plotly_dark"
ACCENT = "#38bdf8"
ACCENT2 = "#818cf8"
ACCENT3 = "#34d399"
WARN = "#f59e0b"

BASE_LAYOUT = dict(
    template=TEMPLATE,
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(family="Inter, sans-serif", size=12, color="#e2e8f0"),
    margin=dict(t=50, b=40, l=50, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)


def histogram(
    series: pd.Series,
    col_name: str,
    nbins: int = 30,
    title: Optional[str] = None,
) -> go.Figure:
    fig = px.histogram(
        series.dropna(),
        nbins=nbins,
        title=title or f"Distribution of {col_name}",
        labels={"value": col_name, "count": "Count"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(**BASE_LAYOUT)
    fig.update_traces(opacity=0.85)
    return fig


def box_plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    kwargs = dict(
        y=value_col,
        title=title or f"Box Plot: {value_col}",
        color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3],
    )
    if group_col:
        kwargs["x"] = group_col
        kwargs["color"] = group_col

    fig = px.box(df.dropna(subset=[value_col]), **kwargs)
    fig.update_layout(**BASE_LAYOUT)
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    add_trendline: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    kwargs = dict(
        x=x_col, y=y_col,
        title=title or f"{y_col} vs {x_col}",
        opacity=0.65,
    )
    if color_col:
        kwargs["color"] = color_col

    if add_trendline:
        kwargs["trendline"] = "ols"

    fig = px.scatter(df.dropna(subset=[x_col, y_col]), **kwargs, color_discrete_sequence=[ACCENT])
    fig.update_layout(**BASE_LAYOUT)
    return fig


def bar_chart(
    series: pd.Series,
    col_name: str,
    top_n: int = 20,
    title: Optional[str] = None,
    orientation: str = "v",
) -> go.Figure:
    counts = series.value_counts().head(top_n)

    if orientation == "h":
        fig = px.bar(
            x=counts.values, y=counts.index.astype(str),
            orientation="h",
            title=title or f"Top {top_n}: {col_name}",
            labels={"x": "Count", "y": col_name},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(**BASE_LAYOUT, yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            x=counts.index.astype(str), y=counts.values,
            title=title or f"Top {top_n}: {col_name}",
            labels={"x": col_name, "y": "Count"},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(**BASE_LAYOUT)

    return fig


def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        aspect="auto",
    )
    fig.update_layout(**BASE_LAYOUT, coloraxis_colorbar=dict(title="r"))
    return fig


def line_chart(
    series: pd.Series,
    title: str = "Trend",
    y_label: str = "Value",
    show_markers: bool = True,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines+markers" if show_markers else "lines",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5),
        name=y_label,
    ))
    fig.update_layout(**BASE_LAYOUT, title=title, xaxis_title="Period", yaxis_title=y_label)
    return fig


def grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg: str = "mean",
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    agg_df = df.groupby(group_col)[value_col].agg(agg).reset_index()
    agg_df = agg_df.sort_values(value_col, ascending=False).head(top_n)

    fig = px.bar(
        agg_df, x=group_col, y=value_col,
        title=title or f"{agg.title()} of {value_col} by {group_col}",
        color_discrete_sequence=[ACCENT],
        text_auto=".2s",
    )
    fig.update_layout(**BASE_LAYOUT)
    return fig


def missing_heatmap(df: pd.DataFrame, title: str = "Missing Values Heatmap") -> go.Figure:
    """Visualise missingness across columns."""
    miss_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]

    if miss_pct.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected 🎉", showarrow=False, font=dict(size=14))
        fig.update_layout(**BASE_LAYOUT, title=title)
        return fig

    colors = miss_pct.map(lambda v: WARN if v > 50 else (ACCENT2 if v > 10 else ACCENT))

    fig = go.Figure(go.Bar(
        x=miss_pct.index,
        y=miss_pct.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in miss_pct.values],
        textposition="outside",
    ))
    fig.update_layout(**BASE_LAYOUT, title=title, xaxis_title="Column", yaxis_title="Missing %")
    return fig


def time_series_with_pct_change(
    ts: pd.Series,
    pct_change: pd.Series,
    title: str = "Trend with Period Change",
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[title, "Period-over-Period Change (%)"])

    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values, mode="lines+markers",
        line=dict(color=ACCENT, width=2), name="Value"
    ), row=1, col=1)

    colors = [ACCENT3 if v >= 0 else WARN for v in pct_change.fillna(0)]
    fig.add_trace(go.Bar(
        x=pct_change.index, y=pct_change.values, marker_color=colors, name="% Change"
    ), row=2, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        showlegend=False,
        height=500,
    )
    return fig


def regression_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    slope: float,
    intercept: float,
    r_squared: float,
    title: Optional[str] = None,
) -> go.Figure:
    clean = df[[x_col, y_col]].dropna()
    x_range = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    y_pred = slope * x_range + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean[x_col], y=clean[y_col],
        mode="markers", name="Data",
        marker=dict(color=ACCENT, opacity=0.6, size=6),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred,
        mode="lines", name=f"OLS fit (R²={r_squared:.3f})",
        line=dict(color=WARN, width=2, dash="dash"),
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=title or f"Regression: {y_col} ~ {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def quality_score_gauge(score: float) -> go.Figure:
    color = ACCENT3 if score >= 80 else (WARN if score >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Data Quality Score", "font": {"size": 14, "color": "#e2e8f0"}},
        number={"suffix": "/100", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [0, 50], "color": "#1e293b"},
                {"range": [50, 80], "color": "#1e293b"},
                {"range": [80, 100], "color": "#1e293b"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig.update_layout(paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"), height=250, margin=dict(t=20, b=0))
    return fig
