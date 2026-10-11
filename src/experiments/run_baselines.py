#!/usr/bin/env python
"""
Run centralized and local-only baseline experiments.
Run: python -m src.experiments.run_baselines --config configs/baselines.yaml
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
from src.models.baselines import run_baseline_experiments, BaselineModels


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline experiments')
    parser.add_argument('--config', type=str, default='configs/baselines.yaml', help='Config file path')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--models', type=str, nargs='+', help='Specific models to run')
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


def main():
    args = parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    n_seeds = config_dict.get('n_seeds', args.n_seeds)
    
    # Load processed data
    print(f"Loading processed data from {args.data_dir}...")
    splits = load_processed_splits(args.data_dir)
    
    print(f"Loaded splits: {list(splits.keys())}")
    for name, split in splits.items():
        if 'X_train' in split:
            print(f"  {name}: train={len(split['X_train'])}, test={len(split['X_test'])}")
    
    # Run experiments
    print(f"\nRunning baseline experiments with {n_seeds} seeds...")
    results_df = run_baseline_experiments(
        splits=splits,
        n_seeds=n_seeds,
        output_dir=args.output_dir,
    )
    
    print(f"\nResults saved to {args.output_dir}/baselines.csv")
    print(f"Summary saved to {args.output_dir}/baselines_summary.csv")
    
# Print final summary
    print("\n=== Final Summary (mean ± std over seeds) ===")
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in ['seed']]
    
    for setting in ['centralized', 'local_only']:
        setting_df = results_df[results_df['setting'] == setting]
        if len(setting_df) == 0:
            continue
        print(f"\n--- {setting.upper()} ---")
        for model_type in setting_df['model_type'].unique():
            model_df = setting_df[setting_df['model_type'] == model_type]
            for col in metric_cols:
                mean_val = model_df[col].mean()
                std_val = model_df[col].std()
                print(f"  {model_type}: {col} = {mean_val:.4f} ± {std_val:.4f}")


if __name__ == '__main__':
    main()