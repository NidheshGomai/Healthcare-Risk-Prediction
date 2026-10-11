#!/usr/bin/env python
"""
Run feature scaling study on gradient descent convergence.
Run: python -m src.experiments.run_scaling --config configs/scaling.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.data.loader import DataLoader
from src.data.cohort import CohortConfig
from src.models.gradient_descent import (
    run_scaling_study,
    run_federated_scaling_comparison,
    create_loss_surface_plot_data,
    GradientDescentRegressor,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run feature scaling study')
    parser.add_argument('--config', type=str, default='configs/scaling.yaml', help='Config file path')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--n-iterations', type=int, default=1000, help='Max GD iterations')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and combine all processed data for centralized scaling study."""
    data_path = Path(data_dir)
    
    # Combine all splits
    all_X = []
    all_y = []
    
    for split_dir in data_path.iterdir():
        if split_dir.is_dir():
            X_train = pd.read_parquet(split_dir / 'X_train.parquet')
            y_train = pd.read_parquet(split_dir / 'y_train.parquet')
            X_test = pd.read_parquet(split_dir / 'X_test.parquet')
            y_test = pd.read_parquet(split_dir / 'y_test.parquet')
            
            all_X.append(X_train)
            all_y.append(y_train.iloc[:, 0].values)
            all_X.append(X_test)
            all_y.append(y_test.iloc[:, 0].values)
    
    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    
    # Select only numeric features for gradient descent
    X_numeric = X.select_dtypes(include=[np.number])
    
    return X_numeric.values, y


def load_splits(data_dir: str) -> dict:
    """Load all processed splits for federated comparison."""
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


def plot_convergence_curves(results_df: pd.DataFrame, output_dir: Path):
    """Plot convergence curves for different scaling methods and learning rates."""
    output_dir = Path(output_dir)
    
    # Filter successful runs
    success_df = results_df[results_df['error'].isna()].copy()
    
    # Plot 1: Loss curves by scaling method for each learning rate
    for lr in sorted(success_df['learning_rate'].unique()):
        lr_df = success_df[success_df['learning_rate'] == lr]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, scaling in enumerate(['none', 'minmax', 'zscore']):
            ax = axes[i]
            scaling_df = lr_df[lr_df['scaling'] == scaling]
            
            for _, row in scaling_df.iterrows():
                if 'loss_history' in row and row['loss_history'] is not None:
                    losses = row['loss_history']
                    batch_type = row['batch_type']
                    label = f"{batch_type} (iter={row['iterations_run']})"
                    ax.plot(losses, label=label, alpha=0.7)
            
            ax.set_title(f'{scaling.capitalize()} Scaling (lr={lr:.0e})')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('MSE Loss')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'convergence_lr_{lr:.0e}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Final test MSE vs learning rate for each scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scaling in ['none', 'minmax', 'zscore']:
        scaling_df = success_df[success_df['scaling'] == scaling]
        batch_gd = scaling_df[scaling_df['batch_type'] == 'batch']
        
        if len(batch_gd) > 0:
            ax.plot(batch_gd['learning_rate'], batch_gd['test_mse'], 
                   marker='o', label=f'{scaling.capitalize()}', alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test MSE (Batch GD)')
    ax.set_title('Test MSE vs Learning Rate by Scaling Method')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mse_vs_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Iterations to convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scaling in ['none', 'minmax', 'zscore']:
        scaling_df = success_df[success_df['scaling'] == scaling]
        batch_gd = scaling_df[scaling_df['batch_type'] == 'batch']
        
        if len(batch_gd) > 0:
            ax.plot(batch_gd['learning_rate'], batch_gd['iterations_run'], 
                   marker='s', label=f'{scaling.capitalize()}', alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Iterations to Convergence vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'iterations_vs_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Divergence heatmap
    pivot = success_df.pivot_table(
        index='learning_rate', 
        columns='scaling', 
        values='test_mse',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Test MSE: Learning Rate vs Scaling (Batch GD)')
    ax.set_xlabel('Scaling Method')
    ax.set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mse_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_surface(X: np.ndarray, y: np.ndarray, output_dir: Path):
    """Create loss surface contour plots for scaled vs unscaled data."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Select two features with highest variance
    variances = np.var(X, axis=0)
    top2_idx = np.argsort(variances)[-2:]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (scaling_name, scaler) in enumerate([
        ('No Scaling', None),
        ('Min-Max', MinMaxScaler()),
        ('Z-Score', StandardScaler()),
    ]):
        if scaler is not None:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create loss surface data
        surface_data = create_loss_surface_plot_data(
            X_scaled, y, feature_indices=(top2_idx[0], top2_idx[1])
        )
        
        w1_range = surface_data['w1_range']
        w2_range = surface_data['w2_range']
        loss_grid = surface_data['loss_grid']
        
        # Contour plot
        W1, W2 = np.meshgrid(w1_range, w2_range)
        contour = axes[i].contourf(W1, W2, loss_grid.T, levels=20, cmap='RdYlGn_r')
        
        # Mark optimal point
        opt_theta = surface_data['optimal_theta']
        axes[i].plot(opt_theta[top2_idx[0] + 1], opt_theta[top2_idx[1] + 1], 
                    'r*', markersize=15, label='Optimal')
        
        axes[i].set_title(scaling_name)
        axes[i].set_xlabel(f'Weight {top2_idx[0]}')
        axes[i].set_ylabel(f'Weight {top2_idx[1]}')
        axes[i].legend()
        
        plt.colorbar(contour, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_surfaces.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_federated_scaling_comparison(fed_results: Dict, output_dir: Path):
    """Plot local vs federated scaling comparison."""
    local = fed_results['local_scaling']
    federated = fed_results['federated_scaling']
    
    clients = list(local.keys())
    x = np.arange(len(clients))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MSE comparison
    local_mse = [local[c]['test_mse'] for c in clients]
    fed_mse = [federated[c]['test_mse'] for c in clients]
    
    axes[0].bar(x - width/2, local_mse, width, label='Local Scaling', alpha=0.8)
    axes[0].bar(x + width/2, fed_mse, width, label='Federated Scaling', alpha=0.8)
    axes[0].set_xlabel('Client')
    axes[0].set_ylabel('Test MSE')
    axes[0].set_title('Test MSE: Local vs Federated Scaling')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clients)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R2 comparison
    local_r2 = [local[c]['test_r2'] for c in clients]
    fed_r2 = [federated[c]['test_r2'] for c in clients]
    
    axes[1].bar(x - width/2, local_r2, width, label='Local Scaling', alpha=0.8)
    axes[1].bar(x + width/2, fed_r2, width, label='Federated Scaling', alpha=0.8)
    axes[1].set_xlabel('Client')
    axes[1].set_ylabel('Test R²')
    axes[1].set_title('Test R²: Local vs Federated Scaling')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clients)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'federated_scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    set_global_seed(config.get('random_state', GLOBAL_SEED))
    
    print("Loading processed data...")
    X, y = load_processed_data(args.data_dir)
    print(f"Data shape: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Load splits for federated comparison
    splits = load_splits(args.data_dir)
    
    # Configuration from config file
    learning_rates = config.get('learning_rates', [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    scaling_methods = config.get('scaling_methods', ['none', 'minmax', 'zscore'])
    batch_sizes = config.get('batch_sizes', [None, 32, 64])
    n_iterations = config.get('n_iterations', args.n_iterations)
    random_state = config.get('random_state', GLOBAL_SEED)
    
    print(f"\nRunning scaling study...")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Scaling methods: {scaling_methods}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Max iterations: {n_iterations}")
    
    # Run scaling study
    results_df = run_scaling_study(
        X=X,
        y=y,
        learning_rates=learning_rates,
        scaling_methods=scaling_methods,
        n_iterations=n_iterations,
        batch_sizes=batch_sizes,
        random_state=random_state,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'scaling_study.csv', index=False)
    print(f"\nResults saved to {output_dir}/scaling_study.csv")
    
    # Generate plots
    print("Generating convergence plots...")
    plot_convergence_curves(results_df, output_dir)
    
    print("Generating loss surface plots...")
    plot_loss_surface(X, y, output_dir)
    
    print("Running federated scaling comparison...")
    fed_results = run_federated_scaling_comparison(
        splits=splits,
        learning_rate=config.get('fed_learning_rate', 0.01),
        n_iterations=n_iterations,
        random_state=random_state,
    )
    
    print("Generating federated scaling comparison plots...")
    plot_federated_scaling_comparison(fed_results, output_dir)
    
    # Save federated comparison results
    fed_df_rows = []
    for method in ['local_scaling', 'federated_scaling']:
        for client, metrics in fed_results[method].items():
            row = {'method': method, 'client': client}
            row.update(metrics)
            fed_df_rows.append(row)
    fed_df = pd.DataFrame(fed_df_rows)
    fed_df.to_csv(output_dir / 'federated_scaling_comparison.csv', index=False)
    
    # Print summary
    print("\n=== Scaling Study Summary ===")
    success_df = results_df[results_df['error'].isna()]
    batch_gd = success_df[success_df['batch_type'] == 'batch']
    
    for scaling in scaling_methods:
        sdf = batch_gd[batch_gd['scaling'] == scaling]
        if len(sdf) > 0:
            best = sdf.loc[sdf['test_mse'].idxmin()]
            print(f"\n{scaling.capitalize()} Scaling (Batch GD):")
            print(f"  Best LR: {best['learning_rate']:.0e}")
            print(f"  Test MSE: {best['test_mse']:.4f}")
            print(f"  Test R²: {best['test_r2']:.4f}")
            print(f"  Iterations: {best['iterations_run']}")
            print(f"  Converged: {best['converged']}")
            print(f"  MSE diff from sklearn: {best['mse_diff']:.2e}")
    
    print("\n=== Federated Scaling Comparison ===")
    for client in fed_results['local_scaling'].keys():
        local_mse = fed_results['local_scaling'][client]['test_mse']
        fed_mse = fed_results['federated_scaling'][client]['test_mse']
        diff = local_mse - fed_mse
        print(f"  {client}: Local MSE={local_mse:.4f}, Fed MSE={fed_mse:.4f}, Diff={diff:.4f}")
    
    print("\nAll figures saved to", output_dir)


if __name__ == '__main__':
    main()