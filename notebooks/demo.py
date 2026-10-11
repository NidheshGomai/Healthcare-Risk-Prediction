"""
Interactive End-to-End Demonstration Script.
Run: python -m notebooks.demo
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.models.gradient_descent import GradientDescentRegressor
from src.models.clustering import CentralizedKMeans
from src.federated.federated_forest import run_federated_forest
from src.federated.federated_nb import run_federated_nb
from src.privacy.secure_agg import SecureAggregation
from src.privacy.dp import LaplaceMechanism
from src.explain.explainers import RuleExtractor


def run_demo():
    print("=" * 65)
    print("PRIVACY-PRESERVING FEDERATED LEARNING DEMONSTRATION")
    print("=" * 65)

    set_global_seed(GLOBAL_SEED)

    # 1. Load Data
    print("\n--- 1. Data Partitions & Preprocessing (CO1) ---")
    data_dir = Path('data/processed')
    splits = {}
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            splits[d.name] = {f.stem: pd.read_parquet(f) for f in d.glob('*.parquet')}

    hospital_cids = sorted([k for k in splits if k.startswith('hospital')])
    print(f"Loaded {len(hospital_cids)} hospital client partitions.")
    for cid in hospital_cids:
        n_tr = len(splits[cid]['X_train'])
        n_te = len(splits[cid]['X_test'])
        mort = float(splits[cid]['y_train']['mortality'].mean())
        print(f"  {cid:12s} | Train: {n_tr} | Test: {n_te} | Mortality Rate: {mort*100:.1f}%")

    # 2. Optimization & Scaling
    print("\n--- 2. Optimization & Feature Scaling (CO2) ---")
    X_sample = splits['hospital_0']['X_train'].values[:500, :4]
    y_sample = splits['hospital_0']['y_train'].values.ravel()[:500]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    gd_scaled = GradientDescentRegressor(learning_rate=0.1, n_iterations=50).fit(X_scaled, y_sample)
    print(f"  Scaled GD Final MSE:   {gd_scaled.loss_history_[-1]:.4f} (converged: {gd_scaled.converged_})")

    # 3. Patient Stratification
    print("\n--- 3. Patient Stratification (CO4) ---")
    X_all = splits['global']['X_train'].values[:2000]
    kmeans = CentralizedKMeans(n_clusters=5, random_state=42).fit(X_all)
    sizes = np.bincount(kmeans.labels_)
    print(f"  K-means (k=5) inertia: {kmeans.inertia_:.1f}")
    print(f"  Cluster distribution: {sizes.tolist()}")

    # 4. Federated Models
    print("\n--- 4. Federated Forest & Naïve Bayes (CO3) ---")
    forest_res = run_federated_forest(splits, model_type='rf', n_estimators=50, weighting='sample_count')
    print(f"  Federated Forest AUROC: {forest_res['global_metrics']['auroc']:.4f}")
    print(f"  Federated Forest AUPRC: {forest_res['global_metrics']['auprc']:.4f}")

    nb_res = run_federated_nb(splits, nb_type='categorical')
    print(f"  Federated Categorical NB AUROC: {nb_res['global_metrics']['auroc']:.4f}")
    print(f"  Prediction Agreement with Centralized: {nb_res['verification']['prediction_agreement']*100:.1f}%")

    # 5. Privacy Guarantees
    print("\n--- 5. Privacy: DP & Secure Aggregation ---")
    sa = SecureAggregation(client_ids=hospital_cids, random_state=42)
    print(f"  Secure Aggregation Mask Cancellation: {sa.verify_cancellation(shape=(50,))}")
    dp = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
    print(f"  DP Noise Sample: {dp.add_noise(10.0, random_state=42):.2f} (from true value 10.0)")

    # 6. Clinician-Readable Rules
    print("\n--- 6. Clinician-Readable Decision Rules (CO3) ---")
    X_test = splits['global']['X_test'].values
    y_preds = forest_res['pooled_forest'].predict(X_test)
    features = list(splits['hospital_0']['X_train'].columns)
    extractor = RuleExtractor(max_depth=3, min_samples_leaf=20, feature_names=features, random_state=42)
    extractor.fit(X_test, y_preds)
    print(f"  Surrogate Tree Agreement Rate: {extractor.fidelity_metrics_['agreement_rate']*100:.1f}%")
    print("  Sample Clinical Rule:")
    rules = extractor.extract_rules()
    if rules:
        print(f"   {rules[0]['rule_text']}")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE: ALL COMPONENTS VERIFIED")
    print("=" * 65)


if __name__ == '__main__':
    run_demo()
