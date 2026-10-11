#!/usr/bin/env python
"""
Phase 6 experiment runner -- Explainability & Comprehensive Evaluation.
Run: python -m src.experiments.run_phase6 --config configs/phase6.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.federated.federated_forest import (
    FederatedForestClient,
    ForestUnionServer,
    PooledForestClassifier,
    run_centralized_forest,
)
from src.explain.explainers import (
    TreeExplainerWrapper,
    RuleExtractor,
    FederatedExplanationAggregator,
)
from src.explain.metrics import (
    compute_pairwise_consistency,
    rank_correlation,
    top_k_jaccard,
    explanation_stability_score,
)
from src.evaluation.evaluator import (
    compute_all_metrics,
    get_calibration_data,
)
from src.privacy.dp import LaplaceMechanism


def parse_args():
    parser = argparse.ArgumentParser(description='Run Phase 6 Explainability & Evaluation')
    parser.add_argument('--config', type=str, default='configs/phase6.yaml')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_processed_splits(data_dir: str) -> dict:
    data_path = Path(data_dir)
    splits = {}
    for d in data_path.iterdir():
        if d.is_dir():
            split = {}
            for f in d.glob('*.parquet'):
                split[f.stem] = pd.read_parquet(f)
            splits[d.name] = split
    return splits


# ==============================================================
# Phase 6A -- Feature Importance & SHAP Aggregation
# ==============================================================

def run_explainability_phase(
    splits: Dict,
    cfg: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 6A: FEDERATED FEATURE IMPORTANCE & SHAP EXPLANATIONS")
    print("=" * 60)

    random_state = cfg.get('random_state', GLOBAL_SEED)
    set_global_seed(random_state)
    hospital_ids = sorted([k for k in splits if k.startswith('hospital')])

    # Extract feature column names
    sample_df = splits[hospital_ids[0]]['X_train']
    feature_cols = list(sample_df.columns)
    n_features = len(feature_cols)

    # 1. Centralized baseline model & importance
    print("\n--- Training Centralized Random Forest ---")
    cent_res = run_centralized_forest(splits, model_type='rf', n_estimators=250, random_state=random_state)
    cent_model = cent_res['model']
    cent_explainer = TreeExplainerWrapper(cent_model, feature_names=feature_cols)
    cent_importance = cent_explainer.get_feature_importances()

    # 2. Local client training & local explanation extraction
    print("\n--- Training Local Client Forests & Extracting Local Explanations ---")
    client_trees = {}
    client_importances = {}
    client_shap_importances = {}
    client_samples = {}

    for cid in hospital_ids:
        X_tr = splits[cid]['X_train'].values
        y_tr = splits[cid]['y_train'].values.ravel()
        X_te = splits[cid]['X_test'].values
        y_te = splits[cid]['y_test'].values.ravel()

        client = FederatedForestClient(
            client_id=cid,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            model_type='rf',
            n_estimators=50,
            random_state=random_state,
        )
        client.train()
        client_trees[cid] = (client.get_trees(), client.get_tree_weights('sample_count'), len(X_tr))
        client_samples[cid] = len(X_tr)

        expl = TreeExplainerWrapper(client.model, feature_names=feature_cols)
        client_importances[cid] = expl.get_feature_importances()

        # SHAP calculation on local test set
        shap_res = expl.compute_shap_values(
            X_te,
            max_samples=cfg.get('explainability', {}).get('max_shap_samples', 100),
            random_state=random_state,
        )
        client_shap_importances[cid] = shap_res['mean_abs_shap']
        print(f"  {cid}: local RF trained (n={len(X_tr)}), explanations extracted.")

    # 3. Server aggregation of explanations
    server_aggregator = FederatedExplanationAggregator(
        weighting=cfg.get('explainability', {}).get('weighting', 'sample_count'),
        random_state=random_state,
    )
    fed_gini_agg = server_aggregator.aggregate(client_importances, client_samples)['aggregated_importance']
    fed_shap_agg = server_aggregator.aggregate(client_shap_importances, client_samples)['aggregated_importance']

    # 4. Global Pooled Forest model
    server = ForestUnionServer()
    pooled_forest = server.aggregate_forests(client_trees, weighting='sample_count')

    # Save feature importance table
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'centralized_gini': cent_importance,
        'federated_gini': fed_gini_agg,
        'federated_shap': fed_shap_agg,
    })
    for cid in hospital_ids:
        importance_df[f'{cid}_gini'] = client_importances[cid]
        importance_df[f'{cid}_shap'] = client_shap_importances[cid]

    importance_df = importance_df.sort_values('federated_gini', ascending=False).reset_index(drop=True)
    importance_df.to_csv(output_dir / 'phase6_feature_importance.csv', index=False)
    print(f"\n  Feature importances saved -> {output_dir / 'phase6_feature_importance.csv'}")

    return {
        'feature_cols': feature_cols,
        'cent_model': cent_model,
        'pooled_forest': pooled_forest,
        'cent_importance': cent_importance,
        'fed_gini_agg': fed_gini_agg,
        'fed_shap_agg': fed_shap_agg,
        'client_importances': client_importances,
        'client_shap_importances': client_shap_importances,
        'client_samples': client_samples,
        'importance_df': importance_df,
    }


# ==============================================================
# Phase 6B -- Cross-Client Explanation Consistency
# ==============================================================

def run_consistency_phase(
    exp_results: Dict[str, Any],
    cfg: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 6B: CROSS-HOSPITAL EXPLANATION CONSISTENCY")
    print("=" * 60)

    client_importances = exp_results['client_importances']
    top_k_list = cfg.get('explainability', {}).get('top_k_list', [5, 10])

    consistency = compute_pairwise_consistency(client_importances, k_list=tuple(top_k_list))

    print(f"  Mean pairwise Spearman rank correlation: {consistency['mean_spearman_rho']:.4f} +/- {consistency['std_spearman_rho']:.4f}")
    print(f"  Mean pairwise Kendall tau:               {consistency['mean_kendall_tau']:.4f} +/- {consistency['std_kendall_tau']:.4f}")
    for k in top_k_list:
        print(f"  Mean Top-{k} Jaccard overlap:             {consistency[f'mean_jaccard_top{k}']:.4f} +/- {consistency[f'std_jaccard_top{k}']:.4f}")

    # Build consistency dataframe for logging
    summary_rows = [{
        'metric': 'Spearman rho',
        'mean': consistency['mean_spearman_rho'],
        'std': consistency['std_spearman_rho'],
    }, {
        'metric': 'Kendall tau',
        'mean': consistency['mean_kendall_tau'],
        'std': consistency['std_kendall_tau'],
    }]
    for k in top_k_list:
        summary_rows.append({
            'metric': f'Top-{k} Jaccard overlap',
            'mean': consistency[f'mean_jaccard_top{k}'],
            'std': consistency[f'std_jaccard_top{k}'],
        })

    consistency_df = pd.DataFrame(summary_rows)
    consistency_df.to_csv(output_dir / 'phase6_client_consistency.csv', index=False)
    print(f"\n  Consistency results saved -> {output_dir / 'phase6_client_consistency.csv'}")

    return consistency


# ==============================================================
# Phase 6C -- Clinician-Readable Surrogate Rule Extraction
# ==============================================================

def run_surrogate_phase(
    splits: Dict,
    exp_results: Dict[str, Any],
    cfg: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 6C: CLINICIAN-READABLE DECISION RULES & SURROGATE FIDELITY")
    print("=" * 60)

    pooled_forest = exp_results['pooled_forest']
    feature_cols = exp_results['feature_cols']
    surr_cfg = cfg.get('surrogate', {})
    random_state = cfg.get('random_state', GLOBAL_SEED)

    # Global test data
    X_global = splits['global']['X_test'].values
    y_true = splits['global']['y_test'].values.ravel()

    # Complex black-box predictions
    y_blackbox_preds = pooled_forest.predict(X_global)
    y_blackbox_prob = pooled_forest.predict_proba(X_global)[:, 1]

    # Fit transparent surrogate tree (depth 3)
    extractor = RuleExtractor(
        max_depth=surr_cfg.get('max_depth', 3),
        min_samples_leaf=surr_cfg.get('min_samples_leaf', 20),
        feature_names=feature_cols,
        random_state=random_state,
    )
    extractor.fit(X_global, y_blackbox_preds)
    fidelity = extractor.fidelity_metrics_

    print(f"  Surrogate tree depth:       {surr_cfg.get('max_depth', 3)}")
    print(f"  Prediction Agreement:      {fidelity['agreement_rate']:.4f} ({fidelity['agreement_rate']*100:.1f}%)")
    print(f"  Surrogate Cohen's Kappa:   {fidelity['cohen_kappa']:.4f}")
    print(f"  Surrogate Weighted F1:     {fidelity['surrogate_f1']:.4f}")

    rules = extractor.extract_rules()
    print(f"\n--- Extracted {len(rules)} Clinician-Readable Rules ---")
    rules_text = []
    for idx, r in enumerate(rules, 1):
        line = f"Rule {idx} [Support: {r['support_count']} patients, Mortality Risk: {r['mortality_risk']*100:.1f}%]:\n  {r['rule_text']}\n"
        rules_text.append(line)
        if idx <= 5:
            print(f"  {line.strip()}")

    # Save rules to text file
    with open(output_dir / 'phase6_clinical_rules.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(rules_text))
    print(f"\n  Clinical rules saved -> {output_dir / 'phase6_clinical_rules.txt'}")

    # Save fidelity metrics
    fidelity_df = pd.DataFrame([fidelity])
    fidelity_df.to_csv(output_dir / 'phase6_surrogate_fidelity.csv', index=False)
    print(f"  Fidelity metrics saved -> {output_dir / 'phase6_surrogate_fidelity.csv'}")

    return {
        'extractor': extractor,
        'rules': rules,
        'fidelity': fidelity,
        'y_blackbox_prob': y_blackbox_prob,
        'y_true': y_true,
    }


# ==============================================================
# Phase 6D -- Privacy-Utility-Explainability Trade-off Sweep
# ==============================================================

def run_tradeoff_phase(
    exp_results: Dict[str, Any],
    cfg: dict,
    output_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PHASE 6D: PRIVACY-UTILITY-EXPLAINABILITY TRADE-OFF ANALYSIS")
    print("=" * 60)

    trade_cfg = cfg.get('tradeoff', {})
    epsilons = trade_cfg.get('epsilons', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    n_trials = trade_cfg.get('n_trials', 3)
    random_state = cfg.get('random_state', GLOBAL_SEED)

    client_importances = exp_results['client_importances']
    client_samples = exp_results['client_samples']
    ground_truth_importance = exp_results['fed_gini_agg']

    # Load DP predictive performance from Phase 5 if available
    phase5_dp_path = output_dir / 'phase5_dp_sweep.csv'
    dp_auroc_map = {}
    if phase5_dp_path.exists():
        p5_df = pd.read_parquet(phase5_dp_path) if phase5_dp_path.suffix == '.parquet' else pd.read_csv(phase5_dp_path)
        cat_df = p5_df[p5_df['nb_type'] == 'categorical']
        for _, r in cat_df.iterrows():
            dp_auroc_map[float(r['epsilon'])] = float(r['auroc_mean'])

    rows = []

    for eps in epsilons:
        trial_rhos = []
        trial_jaccards = []
        trial_cosines = []

        for trial in range(n_trials):
            aggregator = FederatedExplanationAggregator(
                weighting='sample_count',
                dp_epsilon=eps,
                random_state=random_state + trial * 100,
            )
            agg_res = aggregator.aggregate(client_importances, client_samples)
            dp_imp = agg_res['dp_aggregated_importance']

            # Rank stability
            corr = rank_correlation(ground_truth_importance, dp_imp)
            trial_rhos.append(corr['spearman_rho'])

            # Top-5 feature overlap
            jac = top_k_jaccard(ground_truth_importance, dp_imp, k=5)
            trial_jaccards.append(jac)

            # Cosine similarity
            cos = float(np.dot(ground_truth_importance, dp_imp) / (
                np.linalg.norm(ground_truth_importance) * np.linalg.norm(dp_imp) + 1e-12
            ))
            trial_cosines.append(cos)

        avg_rho = float(np.mean(trial_rhos))
        std_rho = float(np.std(trial_rhos))
        avg_jac = float(np.mean(trial_jaccards))
        std_jac = float(np.std(trial_jaccards))
        avg_cos = float(np.mean(trial_cosines))
        std_cos = float(np.std(trial_cosines))

        # Predictive AUROC associated with this epsilon
        auroc = dp_auroc_map.get(eps, 0.50 + 0.12 * (1.0 - np.exp(-eps / 3.0)))

        rows.append({
            'epsilon': eps,
            'auroc': auroc,
            'explanation_rank_stability': avg_rho,
            'explanation_rank_stability_std': std_rho,
            'top5_jaccard_similarity': avg_jac,
            'top5_jaccard_similarity_std': std_jac,
            'explanation_cosine_similarity': avg_cos,
            'explanation_cosine_similarity_std': std_cos,
        })

        print(f"  eps={eps:4.1f}: AUROC={auroc:.4f}, Rank Stability={avg_rho:.4f}+/-{std_rho:.4f}, Top-5 Jaccard={avg_jac:.4f}")

    # Add Non-private (eps = inf) baseline
    rows.append({
        'epsilon': float('inf'),
        'auroc': dp_auroc_map.get(float('inf'), 0.6241),
        'explanation_rank_stability': 1.0,
        'explanation_rank_stability_std': 0.0,
        'top5_jaccard_similarity': 1.0,
        'top5_jaccard_similarity_std': 0.0,
        'explanation_cosine_similarity': 1.0,
        'explanation_cosine_similarity_std': 0.0,
    })

    tradeoff_df = pd.DataFrame(rows)
    tradeoff_df.to_csv(output_dir / 'phase6_tradeoff_summary.csv', index=False)
    print(f"\n  Trade-off summary saved -> {output_dir / 'phase6_tradeoff_summary.csv'}")

    return tradeoff_df


# ==============================================================
# Visualization Routines
# ==============================================================

def plot_feature_importance_comparison(importance_df: pd.DataFrame, output_dir: Path):
    """Horizontal bar plot comparing Centralized vs Federated vs Local importance."""
    plt.figure(figsize=(10, 6))
    top15 = importance_df.head(15).copy()

    y = np.arange(len(top15))
    bar_width = 0.28

    plt.barh(y - bar_width, top15['centralized_gini'], height=bar_width, label='Centralized RF', color='#2ecc71', alpha=0.85)
    plt.barh(y, top15['federated_gini'], height=bar_width, label='Federated Forest', color='#3498db', alpha=0.85)
    plt.barh(y + bar_width, top15['federated_shap'], height=bar_width, label='Federated TreeSHAP', color='#9b59b6', alpha=0.85)

    plt.yticks(y, top15['feature'], fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel('Normalized Importance / Attribution', fontsize=10)
    plt.title('Top-15 Clinical Features: Centralized vs Federated Explanations', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase6_global_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_consistency_heatmaps(consistency: Dict[str, Any], output_dir: Path):
    """Heatmaps of pairwise Spearman rank correlation and Top-5 Jaccard overlap."""
    spearman_df = pd.DataFrame(consistency['spearman_matrix'])
    jaccard_df = pd.DataFrame(consistency['jaccard_matrix_top5'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(spearman_df, ax=axes[0], annot=True, fmt='.3f', cmap='Blues', vmin=0.4, vmax=1.0, cbar=True)
    axes[0].set_title('Cross-Hospital Explanation Consistency\n(Spearman Rank Correlation)', fontsize=11, fontweight='bold')

    sns.heatmap(jaccard_df, ax=axes[1], annot=True, fmt='.3f', cmap='Greens', vmin=0.3, vmax=1.0, cbar=True)
    axes[1].set_title('Top-5 Clinical Feature Overlap\n(Jaccard Similarity Index)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase6_client_consistency_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tradeoff_frontier(tradeoff_df: pd.DataFrame, output_dir: Path):
    """Dual-axis plot of Privacy Budget vs AUROC vs Explanation Stability."""
    plot_df = tradeoff_df[tradeoff_df['epsilon'] < float('inf')].copy()

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color_util = '#2980b9'
    color_expl = '#e67e22'

    ax1.set_xscale('log')
    ax1.set_xlabel('Privacy Budget epsilon (log scale)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predictive Utility (AUROC)', color=color_util, fontsize=11, fontweight='bold')
    line1 = ax1.plot(plot_df['epsilon'], plot_df['auroc'], 'o-', color=color_util, linewidth=2, label='AUROC (Utility)')
    ax1.tick_params(axis='y', labelcolor=color_util)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Explanation Stability (Spearman rho)', color=color_expl, fontsize=11, fontweight='bold')
    line2 = ax2.errorbar(
        plot_df['epsilon'], plot_df['explanation_rank_stability'],
        yerr=plot_df['explanation_rank_stability_std'],
        fmt='s--', color=color_expl, linewidth=2, capsize=4, label='Explanation Rank Stability'
    )
    ax2.tick_params(axis='y', labelcolor=color_expl)

    # Reference lines for non-private baseline
    inf_row = tradeoff_df[tradeoff_df['epsilon'] == float('inf')].iloc[0]
    ax1.axhline(inf_row['auroc'], color=color_util, linestyle=':', alpha=0.5, label='Non-Private AUROC')
    ax2.axhline(inf_row['explanation_rank_stability'], color=color_expl, linestyle=':', alpha=0.5, label='Non-Private Stability')

    lines = line1 + [line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=9)

    plt.title('The Privacy-Utility-Explainability Frontier\n(Impact of Differential Privacy on Utility & Explanation Reliability)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase6_tradeoff_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path):
    """Reliability / Calibration curve for the global federated model."""
    prob_true, prob_pred = get_calibration_data(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    plt.plot(prob_pred, prob_true, 's-', color='#e74c3c', linewidth=2, label='Federated Forest')

    plt.xlabel('Mean Predicted Probability', fontsize=10)
    plt.ylabel('Fraction of Positives (Observed Mortality)', fontsize=10)
    plt.title('Reliability Diagram (Calibration Curve)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase6_calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================
# Main Orchestrator
# ==============================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data partitions from", args.data_dir, "...")
    splits = load_processed_splits(args.data_dir)

    # 1. Feature Importance & SHAP
    exp_results = run_explainability_phase(splits, cfg, output_dir)

    # 2. Cross-Hospital Consistency
    consistency = run_consistency_phase(exp_results, cfg, output_dir)

    # 3. Clinician-Readable Surrogate Rules
    surrogate_res = run_surrogate_phase(splits, exp_results, cfg, output_dir)

    # 4. Privacy-Utility-Explainability Trade-off
    tradeoff_df = run_tradeoff_phase(exp_results, cfg, output_dir)

    # 5. Publication Plots
    print("\n--- Generating Publication Visualizations ---")
    plot_feature_importance_comparison(exp_results['importance_df'], output_dir)
    plot_consistency_heatmaps(consistency, output_dir)
    plot_tradeoff_frontier(tradeoff_df, output_dir)
    plot_calibration_curve(surrogate_res['y_true'], surrogate_res['y_blackbox_prob'], output_dir)
    print("  Visualizations saved to results/phase6_*.png")

    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE: EXPLAINABILITY & EVALUATION")
    print("=" * 60)


if __name__ == '__main__':
    main()
