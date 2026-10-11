# Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction

**Course Project** | **Duration**: 10 weeks | **Language**: Python

## Overview
This project implements a privacy-preserving federated learning framework for healthcare risk prediction (in-hospital mortality / 30-day readmission) across distributed hospital data. The system keeps raw patient data local while enabling collaborative model training with explainability and formal privacy guarantees.

## Key Features
- **Federated Learning**: Horizontal FL across multi-hospital data using Flower
- **Models**: Decision Trees, Random Forest, Naïve Bayes, Linear Regression (scaling study)
- **Privacy**: Differential Privacy (Opacus) and Secure Aggregation
- **Explainability**: SHAP, LIME, feature importance with federated aggregation
- **Clustering**: K-means for patient stratification and client personalization
- **Preprocessing**: Comprehensive pipeline with federated statistics comparison

## Course Outcomes Mapping
| CO | Topic | Phase |
|---|---|---|
| CO1 | Data Preprocessing | Phase 1 |
| CO2 | Feature Scaling & Gradient Descent | Phase 3 |
| CO3 | Classification (DT, RF, NB) | Phases 2, 5 |
| CO4 | K-means Clustering | Phase 4 |

## Repository Structure
```
fl-healthcare/
├── data/                  # raw/ and processed/ (never commit raw data)
├── notebooks/             # EDA, analysis, final figures
├── src/
│   ├── data/              # loading, cohort extraction, partitioning
│   ├── preprocessing/     # imputers, scalers, discretizers, feature engineering
│   ├── models/            # centralized + federated models
│   ├── federated/         # clients, server, strategies, clustering
│   ├── privacy/           # DP and secure aggregation
│   ├── explain/           # SHAP/LIME, consistency metrics
│   └── evaluation/        # metrics, plots
├── configs/               # YAML experiment configs
├── results/               # metrics, figures, logs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Access
Primary: eICU Collaborative Research Database (requires PhysioNet credentialed access)
Fallback: MIMIC-IV Demo (~100 patients) or Synthea synthetic EHR

### 3. Run Preprocessing Pipeline
```bash
python -m src.data.preprocess --config configs/preprocess.yaml
```

### 4. Run Experiments
```bash
# Centralized baselines
python -m src.experiments.run_baselines --config configs/baselines.yaml

# Feature scaling study
python -m src.experiments.run_scaling --config configs/scaling.yaml

# Federated training
python -m src.experiments.run_federated --config configs/federated.yaml
```

## Configuration
All experiments use YAML configs in `configs/`. Example:
```yaml
# configs/preprocess.yaml
dataset: eicu
target: mortality
observation_window: 24  # hours
scaling: zscore
discretization: quantile
n_bins: 10
```

## Results
Outputs saved to `results/`:
- `baselines.csv` - Centralized vs local-only benchmarks
- `scaling_study/` - Convergence plots, loss curves
- `federated/` - FL metrics per round, privacy-utility curves
- `explainability/` - SHAP values, importance consistency metrics
- `figures/` - Publication-ready plots

## Reproducibility
- Global random seed: 42 (set in `src/utils/seeds.py`)
- All experiments configurable via YAML
- Git tracked with data/ in .gitignore

## Documentation
- `PDR_Privacy_Preserving_FL_Healthcare_Risk_Prediction.docx` - Project Design Report
- `Implementation_Plan_Privacy_Preserving_FL.md` - Detailed implementation plan
- `Memory.md` - Running project memory for context

## License
Academic project - not for clinical use.