"""
Overview tab - Project summary, architecture, key metrics.
"""
import streamlit as st
import pandas as pd
from dashboard.data_loader import (
    get_forest_results, get_nb_results, get_dp_sweep,
    get_feature_importance, get_client_consistency
)
from dashboard.components import metric_card, metric_row, render_architecture_diagram, render_co_mapping


def render_overview_tab():
    st.header("📋 Project Overview")
    st.markdown("""
    **Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction**
    
    This project implements a federated learning framework where multiple hospitals collaboratively 
    train models to predict ICU in-hospital mortality **without sharing raw patient data**.
    """)
    
    # Key Metrics Row
    st.subheader("🎯 Key Results at a Glance")
    
    forest_df = get_forest_results()
    nb_df = get_nb_results()
    dp_df = get_dp_sweep()
    consist_df = get_client_consistency()
    
    # Compute best metrics
    best_fed_rf = forest_df[forest_df["setting"].str.contains("federated")].loc[
        forest_df[forest_df["setting"].str.contains("federated")]["auroc"].idxmax()
    ]
    best_fed_nb = nb_df[nb_df["setting"] == "federated"].iloc[0] if len(nb_df[nb_df["setting"] == "federated"]) > 0 else None
    cent_rf = forest_df[forest_df["setting"] == "centralized"].iloc[0]
    local_rf = forest_df[forest_df["setting"] == "local_only_avg"].iloc[0]
    
    rho = consist_df[consist_df["metric"] == "Spearman rho"]["mean"].values[0] if len(consist_df) > 0 else 0
    
    metrics = [
        {"title": "🌲 Best Federated RF AUROC", "value": f"{best_fed_rf['auroc']:.4f}", 
         "delta": f"+{(best_fed_rf['auroc'] - cent_rf['auroc'])*100:.1f}% vs Centralized", "help": "Forest Union (sample_count weighting)"},
        {"title": "🏥 Centralized RF AUROC", "value": f"{cent_rf['auroc']:.4f}", 
         "help": "Upper bound - all data pooled centrally"},
        {"title": "🏠 Local-Only Avg AUROC", "value": f"{local_rf['auroc']:.4f}", 
         "delta": f"{(local_rf['auroc'] - cent_rf['auroc'])*100:.1f}% vs Centralized", "help": "Lower bound - no collaboration"},
        {"title": "🤝 Cross-Hospital Consistency (ρ)", "value": f"{rho:.4f}", 
         "help": "Spearman rank correlation of feature importance across hospitals"},
    ]
    metric_row(metrics, columns=4)
    
    if best_fed_nb is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Federated NB AUROC", f"{best_fed_nb['auroc']:.4f}", 
                     help="Sufficient-statistics aggregation (lossless)")
        with col2:
            # Check if verification exists
            if "pred_agreement" in best_fed_nb and pd.notna(best_fed_nb.get("pred_agreement")):
                st.metric("✅ Prediction Agreement", f"{best_fed_nb['pred_agreement']:.2%}",
                         help="Federated vs Centralized NB")
    
    st.divider()
    
    # Architecture Diagram
    st.subheader("🏗️ System Architecture")
    render_architecture_diagram()
    
    st.divider()
    
    # Course Outcomes Mapping
    st.subheader("🎓 Course Outcomes (CO) Mapping")
    render_co_mapping()
    
    st.divider()
    
    # Dataset Info
    st.subheader("📊 Dataset Characteristics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **Cohort**: Simulated eICU\n
        **Hospitals**: 5\n
        **Task**: In-hospital Mortality (Binary)\n
        **Observation Window**: First 24 hours
        """)
    with col2:
        st.info("""
        **Train Samples**: 8,000\n
        **Test Samples**: 2,000 (global)\n
        **Features**: 38 (demographics, vitals, labs, comorbidities)\n
        **Class Balance**: ~10% mortality
        """)
    with col3:
        st.info("""
        **Partitioning**: Hospital-based (natural non-IID)\n
        **Privacy**: DP (Laplace), Secure Aggregation\n
        **Explainability**: SHAP, Gini, Surrogate Rules\n
        **Reproducibility**: Seed 42, 5 seeds for baselines
        """)
    
    # Quick Links
    st.subheader("🔗 Quick Navigation")
    st.markdown("""
    | Tab | What You'll Find |
    |-----|------------------|
    | **📊 Model Comparison** | Federated vs Centralized vs Local for RF, DT, NB |
    | **🔒 Privacy Analysis** | DP utility curves (ε slider), Secure Aggregation verification |
    | **🧠 Explainability** | Feature importance (SHAP/Gini), Clinical decision rules, Consistency |
    """)


if __name__ == "__main__":
    render_overview_tab()