"""
Main Streamlit Dashboard for Privacy-Preserving Federated Learning Project.
Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Add project root to Python path so dashboard module can be found
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dashboard.tabs.overview import render_overview_tab
from dashboard.tabs.model_comparison import render_model_comparison_tab
from dashboard.tabs.privacy import render_privacy_tab
from dashboard.tabs.explainability import render_explainability_tab


# --- Page Config ---
st.set_page_config(
    page_title="FL Healthcare Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🏥 FL Healthcare</h2>
        <p style="color: #666;">Privacy-Preserving Federated Learning<br/>for ICU Mortality Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📚 Project Info")
    st.markdown("""
    - **Course**: ML Lab Mini Project
    - **COs**: CO1, CO2, CO3, CO4
    - **Dataset**: Simulated eICU (5 hospitals)
    - **Target**: In-hospital Mortality
    - **Seed**: 42 (reproducible)
    """)
    
    st.divider()
    
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [GitHub Repo](https://github.com)
    - [Final Report](reports/final_report.md)
    - [Implementation Plan](Implementation_Plan_Privacy_Preserving_FL.md)
    """)
    
    st.divider()
    
    # Status indicators
    st.markdown("### 📊 Experiment Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baselines", "✅ Done")
        st.metric("Scaling Study", "✅ Done")
    with col2:
        st.metric("Federated Models", "✅ Done")
        st.metric("Explainability", "✅ Done")
    
    st.divider()
    st.caption("Built with Streamlit • Plotly • Pandas")


# --- Main Content ---
st.markdown('<div class="main-header">🏥 Privacy-Preserving Federated Learning for Healthcare Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Dashboard for Model Comparison, Privacy Analysis & Clinical Explainability</div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview",
    "📊 Model Comparison", 
    "🔒 Privacy Analysis",
    "🧠 Explainability"
])

with tab1:
    render_overview_tab()

with tab2:
    render_model_comparison_tab()

with tab3:
    render_privacy_tab()

with tab4:
    render_explainability_tab()

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <small>
    Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction | 
    ML Lab Mini Project | 
    CO1: Preprocessing | CO2: Optimization | CO3: Classification/Privacy | CO4: Clustering
    </small>
</div>
""", unsafe_allow_html=True)