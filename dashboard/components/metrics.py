"""
Metric display components for the dashboard.
"""
import streamlit as st
from typing import List, Dict, Any, Optional


def metric_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Display a metric card with optional delta."""
    st.metric(label=title, value=value, delta=delta, help=help_text)


def metric_row(metrics: List[Dict[str, Any]], columns: int = 4):
    """Display a row of metric cards."""
    cols = st.columns(columns)
    for i, m in enumerate(metrics):
        with cols[i % columns]:
            metric_card(
                title=m.get("title", ""),
                value=m.get("value", ""),
                delta=m.get("delta"),
                help_text=m.get("help")
            )