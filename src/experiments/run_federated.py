#!/usr/bin/env python
"""
Run K-means clustering and Manual Federated Learning experiments.
Run: python -m src.experiments.run_federated --config configs/federated.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.data.loader import DataLoader
from src.data.cohort import CohortConfig
from src.models.clustering import run_kmeans_experiments, find_optimal_k
from src.federated.manual_fl import (
    run_federated_learning,
    run_centralized_baseline,
    run_local_only,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--config', type=str, default='configs/federated.yaml', help='Config file path')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--phase', type=str, choices=['kmeans', 'fl', 'both'], default='both', help='Which phase to run')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_splits(data_dir: str) -> dict:
    """Load processed data splits from parquet files."""
    data_path = Path(data_dir)
    splits = {}
    
    for split_dir in data_path.iterdir():
        if split_dir.is_dir():
            split = {}
            for file in split_dir.glob('*.parquet'):
                key = file.stem
                split[key] = pd.read_parquet(file)
            splits[split_dir.name] = split
    
    return splits


def run_kmeans_phase(splits: Dict, output_dir: str, config: dict):
    """Run K-means clustering experiments."""
    print("\n" + "="*50)
    print("PHASE 4A: K-MEANS CLUSTERING")
    print("="*50)
    
    # Prepare data
    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}
    X_all = np.vstack([v['X_train'].values for v in hospital_splits.values()])
    
    # Find optimal k
    print("\nFinding optimal number of clusters...")
    k_range = config.get('kmeans_k_range', [2, 3, 4, 5, 6, 7, 8])
    elbow_k, sil_k, k_df = find_optimal_k(X_all, k_range)
    print(f"  Elbow method suggests k={elbow_k}")
    print(f"  Silhouette method suggests k={sil_k}")
    
    # Save k selection results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    k_df.to_csv(output_path / 'kmeans_k_selection.csv', index=False)
    
    # Run K-means experiments
    print("\nRunning K-means experiments...")
    kmeans_results = run_kmeans_experiments(
        splits=splits,
        n_clusters_range=k_range,
        output_dir=output_dir,
    )
    
    # Plot k selection
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(k_df['k'], k_df['inertia'], 'o-')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].axvline(elbow_k, color='red', linestyle='--', label=f'Elbow k={elbow_k}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_df['k'], k_df['silhouette'], 'o-', color='orange')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].axvline(sil_k, color='red', linestyle='--', label=f'Best k={sil_k}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'kmeans_k_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return kmeans_results


def run_fl_phase(splits: Dict, output_dir: str, config: dict):
    """Run Manual Federated Learning experiments."""
    print("\n" + "="*50)
    print("PHASE 4B: MANUAL FEDERATED LEARNING (FedAvg)")
    print("="*50)
    
    model_type = config.get('model_type', 'logistic')
    num_rounds = config.get('num_rounds', 10)
    num_clients = config.get('num_clients', 5)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Centralized baseline
    print("\n--- Centralized Baseline ---")
    central_results = run_centralized_baseline(
        splits=splits,
        model_type=model_type,
    )
    print(f"Centralized: acc={central_results['accuracy']:.4f}", end='')
    if 'auroc' in central_results:
        print(f", auroc={central_results['auroc']:.4f}", end='')
    print()
    
    # 2. Local-only baselines
    print("\n--- Local-Only Baselines ---")
    local_results = run_local_only(
        splits=splits,
        model_type=model_type,
    )
    
    # 3. Federated with local scaling
    print("\n--- Federated (Local Scaling) ---")
    fed_local_results = run_federated_learning(
        splits=splits,
        model_type=model_type,
        num_rounds=num_rounds,
        num_clients=num_clients,
        use_global_scaler=False,
        output_dir=output_dir,
    )
    
    # 4. Federated with global scaling
    print("\n--- Federated (Global Scaling) ---")
    fed_global_results = run_federated_learning(
        splits=splits,
        model_type=model_type,
        num_rounds=num_rounds,
        num_clients=num_clients,
        use_global_scaler=True,
        output_dir=output_dir,
    )
    
    # Summary comparison
    print("\n=== FL Results Summary ===")
    print(f"Centralized test: {central_results['accuracy']:.4f}", end='')
    if 'auroc' in central_results:
        print(f", auroc={central_results['auroc']:.4f}", end='')
    print()
    
    local_avg = np.mean([res['accuracy'] for res in local_results.values()])
    for cid, res in local_results.items():
        print(f"Local {cid}: {res['accuracy']:.4f}", end='')
        if 'auroc' in res:
            print(f", auroc={res['auroc']:.4f}", end='')
        print()
    
    fed_local_final = fed_local_results['round_metrics'][-1]['weighted_accuracy'] if fed_local_results['round_metrics'] else None
    fed_global_final = fed_global_results['round_metrics'][-1]['weighted_accuracy'] if fed_global_results['round_metrics'] else None
    
    if fed_local_final:
        print(f"Fed (local scaler): {fed_local_final:.4f}")
    if fed_global_final:
        print(f"Fed (global scaler): {fed_global_final:.4f}")
    
    # Save summary
    summary = {
        'centralized_accuracy': central_results['accuracy'],
        'centralized_auroc': central_results.get('auroc', None),
        'local_tests': {cid: res['accuracy'] for cid, res in local_results.items()},
        'fed_local_final': fed_local_final,
        'fed_global_final': fed_global_final,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / f'fl_{model_type}_summary.csv', index=False)
    
    return {
        'centralized': central_results,
        'local_only': local_results,
        'federated_local_scaler': fed_local_results,
        'federated_global_scaler': fed_global_results,
        'summary': summary,
    }


def plot_fl_comparison(fl_results: Dict, output_dir: str, model_type: str):
    """Plot FL comparison across methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    
    # Round progression
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Local scaler
    if 'federated_local_scaler' in fl_results:
        fed_local = fl_results['federated_local_scaler']
        rounds = [m['round'] for m in fed_local['round_metrics']]
        weighted_acc = [m['weighted_accuracy'] for m in fed_local['round_metrics']]
        axes[0].plot(rounds, weighted_acc, 's-', label='Test (Weighted)')
        axes[0].set_title('Federated (Local Scaler)')
    axes[0].axhline(fl_results['centralized']['accuracy'], color='green', linestyle='--', label='Centralized')
    local_avg = np.mean([res['accuracy'] for res in fl_results['local_only'].values()])
    axes[0].axhline(local_avg, color='red', linestyle='--', label='Local Avg')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Global scaler
    if 'federated_global_scaler' in fl_results:
        fed_global = fl_results['federated_global_scaler']
        rounds = [m['round'] for m in fed_global['round_metrics']]
        weighted_acc = [m['weighted_accuracy'] for m in fed_global['round_metrics']]
        axes[1].plot(rounds, weighted_acc, 's-', label='Test (Weighted)')
        axes[1].set_title('Federated (Global Scaler)')
    axes[1].axhline(fl_results['centralized']['accuracy'], color='green', linestyle='--', label='Centralized')
    axes[1].axhline(local_avg, color='red', linestyle='--', label='Local Avg')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'fl_{model_type}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    set_global_seed(config.get('random_state', GLOBAL_SEED))
    
    print(f"Loading processed data from {args.data_dir}...")
    splits = load_processed_splits(args.data_dir)
    
    print(f"Loaded splits: {list(splits.keys())}")
    for name, split in splits.items():
        if 'X_train' in split:
            print(f"  {name}: train={len(split['X_train'])}, test={len(split['X_test'])}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.phase in ['kmeans', 'both']:
        run_kmeans_phase(splits, args.output_dir, config)
    
    if args.phase in ['fl', 'both']:
        fl_results = run_fl_phase(splits, args.output_dir, config)
        plot_fl_comparison(fl_results, args.output_dir, config.get('model_type', 'logistic'))
    
    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()