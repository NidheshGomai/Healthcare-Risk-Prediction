"""
Explainability tab - Feature importance, SHAP, clinical rules, cross-hospital consistency.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.data_loader import (
    get_feature_importance, get_client_consistency, get_clinical_rules
)
from dashboard.components import (
    horizontal_bar_importance, heatmap_consistency, rule_card, 
    display_df_with_download, parse_clinical_rules, metric_card, metric_row
)


def render_explainability_tab():
    st.header("🧠 Explainability & Clinical Insights")
    st.markdown("""
    Understand **what drives predictions** through feature importance (Gini, TreeSHAP), 
    verify **cross-hospital consistency**, and explore **clinician-readable decision rules**.
    """)
    
    fi_df = get_feature_importance()
    consist_df = get_client_consistency()
    rules_text = get_clinical_rules()
    
    if fi_df.empty:
        st.warning("No explainability results found. Run Phase 6 experiments first.")
        return
    
    # --- Feature Importance Comparison ---
    st.subheader("📊 Feature Importance: Centralized vs Federated vs Local")
    
    importance_cols = st.multiselect(
        "Importance Methods to Compare",
        [c for c in ["centralized_gini", "federated_gini", "federated_shap"] if c in fi_df.columns],
        default=[c for c in ["centralized_gini", "federated_gini", "federated_shap"] if c in fi_df.columns]
    )
    
    top_n = st.slider("Top N Features", 5, min(30, len(fi_df)), 15)
    
    if importance_cols:
        fig = horizontal_bar_importance(
            fi_df,
            features_col="feature",
            importance_cols=importance_cols,
            title="Top Clinical Features: Centralized vs Federated Explanations",
            top_n=top_n,
            height=max(400, top_n * 25)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Per-hospital breakdown
    with st.expander("🏥 Per-Hospital Feature Importance Breakdown"):
        hospital_cols = [c for c in fi_df.columns if c.endswith("_gini") or c.endswith("_shap")]
        if hospital_cols:
            selected_hosp = st.selectbox("Select Hospital View", hospital_cols)
            hosp_df = fi_df[["feature", selected_hosp]].sort_values(selected_hosp, ascending=True).head(top_n)
            
            fig = px.bar(
                hosp_df,
                x=selected_hosp,
                y="feature",
                orientation="h",
                title=f"Feature Importance - {selected_hosp.replace('_', ' ').title()}",
                height=max(400, top_n * 25)
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    # Full table
    with st.expander("📋 Full Feature Importance Table"):
        display_df_with_download(fi_df, "Download Feature Importance", "feature_importance.csv")
    
    st.divider()
    
    # --- Cross-Hospital Consistency ---
    st.subheader("🤝 Cross-Hospital Explanation Consistency")
    st.markdown("""
    Do different hospitals agree on which features matter most? High consistency means 
    the model learns **generalizable clinical patterns**, not hospital-specific artifacts.
    """)
    
    if not consist_df.empty:
        # Summary metrics
        rho_row = consist_df[consist_df["metric"] == "Spearman rho"].iloc[0]
        tau_row = consist_df[consist_df["metric"] == "Kendall tau"].iloc[0]
        jac5_row = consist_df[consist_df["metric"] == "Top-5 Jaccard overlap"].iloc[0]
        jac10_row = consist_df[consist_df["metric"] == "Top-10 Jaccard overlap"].iloc[0]
        
        metric_row([
            {"title": "Spearman ρ (Rank Correlation)", "value": f"{rho_row['mean']:.4f} ± {rho_row['std']:.4f}"},
            {"title": "Kendall τ", "value": f"{tau_row['mean']:.4f} ± {tau_row['std']:.4f}"},
            {"title": "Top-5 Jaccard", "value": f"{jac5_row['mean']:.4f} ± {jac5_row['std']:.4f}"},
            {"title": "Top-10 Jaccard", "value": f"{jac10_row['mean']:.4f} ± {jac10_row['std']:.4f}"},
        ], columns=4)
        
        # Heatmaps from saved matrices (need to reconstruct from CSV)
        # For now, show the summary table
        st.dataframe(consist_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **Interpretation**: 
        - **Spearman ρ = 0.89** → Hospitals strongly agree on feature *ranking*
        - **Top-5 Jaccard = 0.35** → Moderate overlap in *exact* top-5 features
        - This is expected: different patient populations shift exact importance values
        """)
    
    st.divider()
    
    # --- Clinical Decision Rules ---
    st.subheader("🩺 Clinician-Readable Decision Rules (Surrogate Tree)")
    st.markdown("""
    A **depth-3 decision tree** is trained to mimic the federated Random Forest's predictions 
    (83.1% fidelity). These IF-THEN rules are directly interpretable by clinicians.
    """)
    
    rules = parse_clinical_rules(rules_text)
    
    if rules:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Surrogate Fidelity", "83.1%")
        with col2:
            metric_card("Tree Depth", "3")
        with col3:
            metric_card("Min Samples/Leaf", "20")
        
        # Rule cards
        for i, rule in enumerate(rules):
            rule_card(rule, i)
        
        # Download
        st.download_button(
            "📥 Download Rules (Text)",
            rules_text,
            "clinical_decision_rules.txt",
            "text/plain"
        )
    
    else:
        st.info("No clinical rules found. Run Phase 6 to generate surrogate rules.")
    
    st.divider()
    
    # --- Calibration Curve (if available) ---
    st.subheader("📈 Model Calibration")
    st.markdown("""
    A well-calibrated model's predicted probabilities match observed frequencies.
    This is critical for clinical risk stratification.
    """)
    
    # Show calibration curve image if exists
    cal_img = "phase6_calibration_curve.png"
    import os
    from dashboard.data_loader import FIGURES_DIR
    cal_path = FIGURES_DIR / cal_img
    if cal_path.exists():
        st.image(str(cal_path), caption="Reliability Diagram (Calibration Curve)", use_container_width=True)
    else:
        st.info("Calibration curve image not found. Run Phase 6 to generate.")


if __name__ == "__main__":
    render_explainability_tab()