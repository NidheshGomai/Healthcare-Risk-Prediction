#!/usr/bin/env python
"""
Main preprocessing script.
Run: python -m src.data.preprocess --config configs/preprocess.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.data.loader import DataLoader, load_demo_data
from src.data.cohort import CohortConfig
from src.preprocessing.pipeline import build_preprocessing_pipeline, get_feature_names_after_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess healthcare data for FL')
    parser.add_argument('--config', type=str, default='configs/preprocess.yaml', help='Config file path')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--use-demo', action='store_true', help='Use synthetic demo data')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED, help='Random seed')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_feature_types(X: pd.DataFrame) -> tuple:
    """Auto-detect numeric and categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Move binary numeric to categorical
    binary_numeric = [c for c in numeric_features if X[c].nunique() == 2]
    for c in binary_numeric:
        numeric_features.remove(c)
        categorical_features.append(c)
    
    return numeric_features, categorical_features


def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    config_dict = load_config(args.config)
    config = CohortConfig(**{k: v for k, v in config_dict.items() if k in CohortConfig.__dataclass_fields__})
    
    print(f"Loading data (demo={args.use_demo})...")
    
    if args.use_demo:
        loader = DataLoader(config)
        X, y = loader.load_synthetic(n_samples=10000, n_hospitals=5)
    else:
        # TODO: Load from eICU/MIMIC database
        print("Real data loading not implemented yet. Using demo data.")
        loader = DataLoader(config)
        X, y = loader.load_synthetic(n_samples=10000, n_hospitals=5)
    
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Detect feature types
    numeric_features, categorical_features = detect_feature_types(X)
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:10]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}...")
    
    # Override from config if specified
    if config_dict.get('numeric_features'):
        numeric_features = config_dict['numeric_features']
    if config_dict.get('categorical_features'):
        categorical_features = config_dict['categorical_features']
    
    # Build preprocessing pipeline
    pipeline = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaling_method=config_dict.get('scaling_method', 'zscore'),
        discretization_method=config_dict.get('discretization_method', 'quantile'),
        n_bins=config_dict.get('n_bins', 10),
        add_missingness_indicators=config_dict.get('add_missingness_indicators', True),
        clip_outliers=config_dict.get('clip_outliers', True),
        add_temporal_features=config_dict.get('add_temporal_features', False),
        add_comorbidity_count=config_dict.get('add_comorbidity_count', True),
    )
    
    # Fit and transform
    print("Fitting preprocessing pipeline...")
    X_transformed = pipeline.fit_transform(X, y)
    
    feature_names = get_feature_names_after_preprocessing(pipeline, numeric_features, categorical_features)
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Feature names: {feature_names[:20]}...")
    
    # Create federated splits
    partition_method = config_dict.get('partition_method', 'hospital')
    print(f"Creating federated splits using {partition_method}...")
    
    if partition_method == 'hospital':
        splits = loader.prepare_federated_data(X, y, partition_method='hospital')
    else:
        splits = loader.prepare_federated_data(
            X, y,
            partition_method=partition_method,
            n_clients=config_dict.get('n_clients', 5),
            alpha=config_dict.get('dirichlet_alpha', 0.5),
        )
    
    # Apply preprocessing to each split
    processed_splits = {}
    for name, split in splits.items():
        print(f"Processing {name}: train={len(split['X_train'])}, test={len(split['X_test'])}")
        X_train_trans = pipeline.transform(split['X_train'])
        X_test_trans = pipeline.transform(split['X_test'])
        
        processed_splits[name] = {
            'X_train': pd.DataFrame(X_train_trans, columns=feature_names, index=split['X_train'].index),
            'y_train': split['y_train'],
            'X_test': pd.DataFrame(X_test_trans, columns=feature_names, index=split['X_test'].index),
            'y_test': split['y_test'],
        }
    
    # Save
    output_dir = Path(args.output_dir)
    loader.save_processed_data(processed_splits, str(output_dir))
    print(f"Saved processed data to {output_dir}")
    
    # Print summary
    print("\n=== Summary ===")
    for name, split in processed_splits.items():
        pos_rate = split['y_train'].mean()
        print(f"{name}: train={len(split['X_train'])}, test={len(split['X_test'])}, pos_rate={pos_rate:.3f}")
    
    # Save pipeline
    import joblib
    joblib.dump(pipeline, output_dir / 'preprocessing_pipeline.joblib')
    print(f"Saved pipeline to {output_dir}/preprocessing_pipeline.joblib")


if __name__ == '__main__':
    main()