"""
Chart components for the dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np


def bar_chart_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    y_title: str = "",
    text_fmt: str = ".4f",
    height: int = 400
) -> go.Figure:
    """Create a grouped bar chart for model comparisons."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode="group",
        text_auto=text_fmt,
        title=title,
        height=height
    )
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_title or y_col.replace("_", " ").title(),
        legend_title=color_col.replace("_", " ").title() if color_col else "",
        template="plotly_white",
        showlegend=True
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def line_chart_dp(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    error_col: Optional[str] = None,
    title: str = "",
    x_log: bool = True,
    height: int = 400
) -> go.Figure:
    """Create line chart for DP sweep with error bars."""
    fig = go.Figure()
    
    for name, group in df.groupby(color_col):
        group = group.sort_values(x_col)
        x_vals = group[x_col].values
        y_vals = group[y_col].values
        
        if error_col and error_col in group.columns:
            err_vals = group[error_col].values
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=name,
                error_y=dict(type="data", array=err_vals, visible=True),
                line=dict(width=2),
                marker=dict(size=8)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=name,
                line=dict(width=2),
                marker=dict(size=8)
            ))
    
    if x_log:
        fig.update_xaxes(type="log", title=x_col.replace("_", " ").title())
    else:
        fig.update_xaxes(title=x_col.replace("_", " ").title())
    
    fig.update_yaxes(title=y_col.replace("_", " ").title())
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def heatmap_consistency(
    matrix: Dict[str, Dict[str, float]],
    title: str = "",
    colorscale: str = "Blues",
    zmin: float = 0,
    zmax: float = 1,
    height: int = 400
) -> go.Figure:
    """Create heatmap from nested dict matrix."""
    labels = list(matrix.keys())
    z = [[matrix[r][c] for c in labels] for r in labels]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )
    return fig


def horizontal_bar_importance(
    df: pd.DataFrame,
    features_col: str,
    importance_cols: List[str],
    title: str = "",
    top_n: int = 15,
    height: int = 500
) -> go.Figure:
    """Create horizontal bar chart for feature importance comparison."""
    df_top = df.head(top_n).copy()
    df_top = df_top.sort_values(importance_cols[0], ascending=True)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(importance_cols):
        if col in df_top.columns:
            fig.add_trace(go.Bar(
                y=df_top[features_col],
                x=df_top[col],
                orientation="h",
                name=col.replace("_", " ").title(),
                marker_color=colors[i % len(colors)],
                opacity=0.85
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Normalized Importance",
        yaxis_title="Feature",
        height=height,
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def scatter_equivalence(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    height: int = 400
) -> go.Figure:
    """Create scatter plot for equivalence checking (y=x line)."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        height=height
    )
    
    # Add y=x line
    min_val = min(df[x_col].min(), df[y_col].min())
    max_val = max(df[x_col].max(), df[y_col].max())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="black", dash="dash", width=1),
        opacity=0.5
    )
    
    fig.update_layout(template="plotly_white")
    return fig