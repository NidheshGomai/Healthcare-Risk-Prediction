"""
Architecture diagram and CO mapping components.
"""
import streamlit as st
import pandas as pd


def render_architecture_diagram():
    """Render the federated learning architecture as a visual diagram using Streamlit columns."""
    st.markdown("### Federated Learning Architecture")
    
    # Server column
    st.markdown("#### 🖥️ **Federated Server**")
    server_cols = st.columns(4)
    with server_cols[0]:
        st.info("🌲 **Forest Union Pool**\nSoft-Vote Ensemble (250 trees)")
    with server_cols[1]:
        st.info("📊 **NB Sufficient-Stat**\nLossless Reconstruction")
    with server_cols[2]:
        st.info("🔐 **Secure Aggregation**\nPairwise Mask Cancellation")
    with server_cols[3]:
        st.info("🧠 **Federated Explainability**\nTreeSHAP + Surrogate Rules")
    
    st.divider()
    
    # Client columns
    st.markdown("#### 🏥 **Hospital Clients (5 hospitals)**")
    client_cols = st.columns(5)
    
    for i in range(5):
        with client_cols[i]:
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff7f0e; height: 200px;">
                <h4>Hospital {i+1}</h4>
                <ul style="font-size: 0.9rem; line-height: 1.8;">
                    <li>✅ Preprocessing (CO1)</li>
                    <li>🤖 Local Models (CO3)</li>
                    <li>🧠 Local Explanations (CO3)</li>
                    <li>🔐 Pairwise Masks</li>
                </ul>
                <p style="font-size: 0.8rem; color: #666;">→ Trees/Stats/Masked Updates → Server</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Data flow summary
    st.markdown("#### 🔄 **Data Flow Summary**")
    flow_cols = st.columns(3)
    with flow_cols[0]:
        st.markdown("""
        **Model Updates**
        - Local RF/DT trees → Forest Union
        - NB sufficient statistics → Global NB
        - FedAvg weights → Global Logistic
        """)
    with flow_cols[1]:
        st.markdown("""
        **Privacy**
        - DP noise on statistics (ε=5-10)
        - Pairwise masks cancel exactly
        - Max diff: 3.68×10⁻¹⁶
        """)
    with flow_cols[2]:
        st.markdown("""
        **Explanations**
        - Local Gini/SHAP → Aggregated
        - Cross-hospital ρ = 0.89
        - Surrogate rules (83% fidelity)
        """)


def render_co_mapping():
    """Display CO mapping table."""
    co_data = {
        "CO": ["CO1", "CO2", "CO3", "CO4"],
        "Focus": [
            "Data Preprocessing & Cohort Extraction",
            "Optimization & Feature Scaling Study",
            "Supervised Classification & Privacy",
            "Unsupervised Clustering & Subgroups"
        ],
        "Components": [
            "Pipeline, ClinicalClipper, MissingnessIndicator, Non-IID partitioning",
            "From-scratch Batch GD & Mini-batch SGD, Loss surfaces, Scaling comparison",
            "Federated Forest Union, Federated NB, DP (Laplace), Secure Aggregation",
            "Centralized & Federated K-means, Patient subgroups, FedAvg centroids"
        ],
        "Evidence": [
            "15 unit tests passing, `src/preprocessing/`, `src/data/cohort.py`",
            "`src/models/gradient_descent.py`, convergence plots, MSE heatmaps",
            "Forest 0.765 AUROC, NB 100% agreement, ε=5-10 window, max diff 3.68e-16",
            "Optimal k=5, Silhouette=0.044, ARI=0.0027, 20 FL rounds"
        ]
    }
    df = pd.DataFrame(co_data)
    st.dataframe(df, use_container_width=True, hide_index=True)