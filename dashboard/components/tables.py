"""
Table components for the dashboard.
"""
import pandas as pd
from typing import List, Optional
import streamlit as st


def styled_dataframe(df: pd.DataFrame, highlight_cols: Optional[List[str]] = None, precision: int = 4) -> pd.io.formats.style.Styler:
    """Return a styled dataframe for display."""
    styler = df.style.format(precision=precision)
    
    if highlight_cols:
        for col in highlight_cols:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap="RdYlGn")
    
    return styler


def display_df_with_download(df: pd.DataFrame, label: str = "Download CSV", filename: str = "data.csv"):
    """Display dataframe with download button."""
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, filename, "text/csv")