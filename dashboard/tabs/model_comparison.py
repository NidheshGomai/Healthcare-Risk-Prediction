"""
Model Comparison tab - Interactive comparison of Federated vs Centralized vs Local.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.data_loader import (
    get_forest_results, get_nb_results, get_baselines_summary
)
from dashboard.components import (
    bar_chart_comparison, scatter_equivalence, display_df_with_download
)


def render_model_comparison_tab():
    st.header("📊 Model Comparison")
    st.markdown("""
    Compare **Federated** (Forest Union / Sufficient Statistics) against **Centralized** (all data pooled) 
    and **Local-Only** (no collaboration) baselines across multiple model architectures.
    """)
    
    forest_df = get_forest_results()
    nb_df = get_nb_results()
    baselines_df = get_baselines_summary()
    
    if forest_df.empty and nb_df.empty:
        st.warning("No model results found. Run experiments first.")
        return
    
    # --- Model Selector ---
    all_models = []
    if not forest_df.empty:
        all_models.extend([f"Forest: {m}" for m in forest_df["model"].unique()])
    if not nb_df.empty:
        all_models.extend([f"NB: {m}" for m in nb_df["nb_type"].unique()])
    
    selected_model = st.selectbox(
        "Select Model Architecture",
        all_models,
        index=0
    )
    
    is_forest = selected_model.startswith("Forest")
    model_name = selected_model.split(": ")[1]
    
    if is_forest:
        df = forest_df[forest_df["model"] == model_name].copy()
        setting_col = "setting"
        weighting_col = "weighting"
    else:
        df = nb_df[nb_df["nb_type"] == model_name].copy()
        setting_col = "setting"
        weighting_col = None
    
    # --- Metric Selector ---
    metric_cols = [c for c in df.columns if c in ["accuracy", "auroc", "auprc", "brier"]]
    selected_metric = st.selectbox("Metric", metric_cols, index=1 if "auroc" in metric_cols else 0)
    
    # --- Settings Filter ---
    available_settings = df[setting_col].unique().tolist()
    selected_settings = st.multiselect(
        "Settings to Compare",
        available_settings,
        default=[s for s in available_settings if "federated" in s or "centralized" in s or "local" in s]
    )
    
    plot_df = df[df[setting_col].isin(selected_settings)].copy()
    
    if plot_df.empty:
        st.warning("No data for selected settings.")
        return
    
    # --- Bar Chart ---
    st.subheader(f"{selected_model} - {selected_metric.upper()} Comparison")
    
    if weighting_col and weighting_col in plot_df.columns:
        # For forest: show weighting as hue
        fig = bar_chart_comparison(
            plot_df,
            x_col=setting_col,
            y_col=selected_metric,
            color_col=weighting_col,
            title=f"{selected_model} - {selected_metric.upper()} by Setting & Weighting",
            y_title=selected_metric.upper()
        )
    else:
        fig = bar_chart_comparison(
            plot_df,
            x_col=setting_col,
            y_col=selected_metric,
            title=f"{selected_model} - {selected_metric.upper()} by Setting",
            y_title=selected_metric.upper()
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Detailed Table ---
    st.subheader("📋 Detailed Results")
    
    display_cols = [setting_col]
    if weighting_col and weighting_col in plot_df.columns:
        display_cols.append(weighting_col)
    display_cols.extend([c for c in ["accuracy", "auroc", "auprc", "brier", "n_trees"] if c in plot_df.columns])
    
    display_df = plot_df[display_cols].copy()
    for col in ["accuracy", "auroc", "auprc", "brier"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    display_df_with_download(display_df, f"Download {selected_model} Results", f"{model_name.lower()}_comparison.csv")
    
    # --- Centralized vs Federated Equivalence (for NB) ---
    if not is_forest and "setting" in df.columns:
        cent = df[df["setting"] == "centralized"]
        fed = df[df["setting"] == "federated"]
        
        if len(cent) > 0 and len(fed) > 0:
            st.subheader("🔄 Federated vs Centralized Equivalence Check")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Centralized AUROC", f"{cent['auroc'].values[0]:.4f}")
                if "auprc" in cent.columns:
                    st.metric("Centralized AUPRC", f"{cent['auprc'].values[0]:.4f}")
            
            with col2:
                st.metric("Federated AUROC", f"{fed['auroc'].values[0]:.4f}")
                if "auprc" in fed.columns:
                    st.metric("Federated AUPRC", f"{fed['auprc'].values[0]:.4f}")
            
            # Verification metrics if available
            if "pred_agreement" in fed.columns and pd.notna(fed["pred_agreement"].values[0]):
                st.success(f"✅ **Prediction Agreement: {fed['pred_agreement'].values[0]:.2%}**")
            if "prob_rmse" in fed.columns and pd.notna(fed["prob_rmse"].values[0]):
                st.success(f"✅ **Probability RMSE: {fed['prob_rmse'].values[0]:.2e}**")
            
            st.info("""
            **Why this matters**: Federated Naive Bayes aggregates *sufficient statistics* (counts, sums, sums-of-squares). 
            Mathematically, this reconstructs the **exact same model** as centralized training — no information loss.
            """)
    
    # --- Baselines Summary (if available) ---
    if not baselines_df.empty:
        with st.expander("📈 All Baselines Summary (5 seeds, mean ± std)"):
            st.dataframe(baselines_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_model_comparison_tab()