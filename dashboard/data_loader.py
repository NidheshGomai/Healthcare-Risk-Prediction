"""
Data loader for dashboard - loads all pre-computed results from results/ directory.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from results directory."""
    path = RESULTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_image(name: str) -> Optional[str]:
    """Get path to figure image if exists."""
    path = FIGURES_DIR / name
    if path.exists():
        return str(path)
    return None


@st.cache_data
def load_text(name: str) -> str:
    """Load text file from results directory."""
    path = RESULTS_DIR / name
    if path.exists():
        return path.read_text()
    return ""


# --- Specific loaders for each result file ---

def get_forest_results() -> pd.DataFrame:
    return load_csv("phase5_forest_results.csv")


def get_nb_results() -> pd.DataFrame:
    return load_csv("phase5_nb_results.csv")


def get_baselines_summary() -> pd.DataFrame:
    return load_csv("baselines_summary.csv")


def get_dp_sweep() -> pd.DataFrame:
    return load_csv("phase5_dp_sweep.csv")


def get_secagg_results() -> pd.DataFrame:
    return load_csv("phase5_secagg_results.csv")


def get_feature_importance() -> pd.DataFrame:
    return load_csv("phase6_feature_importance.csv")


def get_client_consistency() -> pd.DataFrame:
    return load_csv("phase6_client_consistency.csv")


def get_clinical_rules() -> str:
    return load_text("phase6_clinical_rules.txt")


def get_tradeoff_summary() -> pd.DataFrame:
    return load_csv("phase6_tradeoff_summary.csv")


def get_kmeans_profiles(k: int) -> pd.DataFrame:
    return load_csv(f"kmeans_centralized_k{k}_profiles.csv")


def get_kmeans_comparison() -> pd.DataFrame:
    return load_csv("kmeans_comparison_summary.csv")


def get_kmeans_selection() -> pd.DataFrame:
    return load_csv("kmeans_k_selection.csv")


def get_scaling_study() -> pd.DataFrame:
    return load_csv("scaling_study.csv")


def get_federated_scaling() -> pd.DataFrame:
    return load_csv("federated_scaling_comparison.csv")


def get_figure(name: str) -> Optional[str]:
    return load_image(name)


# --- Utility functions ---

def get_unique_models(df: pd.DataFrame) -> List[str]:
    if "model" in df.columns:
        return sorted(df["model"].unique().tolist())
    if "nb_type" in df.columns:
        return sorted(df["nb_type"].unique().tolist())
    return []


def get_unique_settings(df: pd.DataFrame) -> List[str]:
    if "setting" in df.columns:
        return sorted(df["setting"].unique().tolist())
    return []


def format_metric(val: float, fmt: str = ".4f") -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:{fmt}}"


def parse_clinical_rules(text: str) -> List[Dict[str, Any]]:
    """Parse clinical rules text into structured list."""
    rules = []
    lines = text.strip().split("\n")
    current_rule = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Rule "):
            if current_rule:
                rules.append(current_rule)
            current_rule = {"text": line}
        elif line.startswith("IF"):
            current_rule["condition"] = line
            # Extract risk and support
            import re
            risk_match = re.search(r"Risk:\s*([\d.]+)%", line)
            support_match = re.search(r"Support:\s*(\d+)", line)
            pred_match = re.search(r"Mortality\s*=\s*(\d+)", line)
            if risk_match:
                current_rule["risk"] = float(risk_match.group(1))
            if support_match:
                current_rule["support"] = int(support_match.group(1))
            if pred_match:
                current_rule["prediction"] = int(pred_match.group(1))
    if current_rule:
        rules.append(current_rule)
    return rules