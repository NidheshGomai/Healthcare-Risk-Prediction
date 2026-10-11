#!/usr/bin/env python
"""
EDA Script: Exploratory Data Analysis for Federated Healthcare Risk Prediction
Run: python -m notebooks.eda
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load processed data
processed_dir = Path('data/processed')
pipeline = joblib.load(processed_dir / 'preprocessing_pipeline.joblib')

# Load all splits
splits = {}
for split_dir in processed_dir.iterdir():
    if split_dir.is_dir():
        split = {}
        for file in split_dir.glob('*.parquet'):
            key = file.stem
            split[key] = pd.read_parquet(file)
        splits[split_dir.name] = split

# Combine all training data for global EDA
X_train_all = pd.concat([s['X_train'] for s in splits.values()], ignore_index=True)
y_train_all = pd.concat([s['y_train'] for s in splits.values()], ignore_index=True)

# Combine all test data
X_test_all = pd.concat([s['X_test'] for s in splits.values()], ignore_index=True)
y_test_all = pd.concat([s['y_test'] for s in splits.values()], ignore_index=True)

print(f"Global train: {X_train_all.shape}, test: {X_test_all.shape}")
print(f"Train positive rate: {y_train_all.iloc[:,0].mean():.3f}")
print(f"Test positive rate: {y_test_all.iloc[:,0].mean():.3f}")
print(f"Clients: {len([k for k in splits.keys() if k.startswith('hospital')])}")

# ============================================================
# 1. Missingness Analysis
# ============================================================
missing_train = X_train_all.isnull().sum()
missing_test = X_test_all.isnull().sum()

print("\n=== Missing Values (Train) ===")
if missing_train.sum() > 0:
    print(missing_train[missing_train > 0])
else:
    print("No missing values")

print("\n=== Missing Values (Test) ===")
if missing_test.sum() > 0:
    print(missing_test[missing_test > 0])
else:
    print("No missing values")

# ============================================================
# 2. Class Balance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Global class distribution
y_train_all.iloc[:,0].value_counts().plot(kind='bar', ax=axes[0], title='Global Train Class Distribution', color=['#2ecc71', '#e74c3c'])
axes[0].set_xticklabels(['Survived (0)', 'Died (1)'], rotation=0)
axes[0].set_ylabel('Count')

# Per-client class distribution
client_pos_rates = {}
for name, split in splits.items():
    if name.startswith('hospital') or name == 'global':
        pos_rate = split['y_train'].iloc[:,0].mean()
        client_pos_rates[name] = pos_rate

pd.Series(client_pos_rates).plot(kind='bar', ax=axes[1], title='Positive Rate per Client', color='steelblue')
axes[1].set_ylabel('Mortality Rate')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/class_balance.png', dpi=150, bbox_inches='tight')
plt.close()

print("Per-client positive rates:")
for name, rate in client_pos_rates.items():
    print(f"  {name}: {rate:.3f}")

# ============================================================
# 3. Feature Distributions
# ============================================================
numeric_cols = X_train_all.select_dtypes(include=[np.number]).columns.tolist()

key_features = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp', 
                'respiratory_rate', 'temperature', 'spo2',
                'glucose', 'creatinine', 'lactate', 'sofa_score', 'apache_score']

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    if feat in X_train_all.columns:
        axes[i].hist(X_train_all[feat], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axes[i].set_title(feat)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

# Hide unused subplots
for i in range(len(key_features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Outlier Analysis
# ============================================================
outlier_summary = {}
for feat in numeric_cols:
    Q1 = X_train_all[feat].quantile(0.25)
    Q3 = X_train_all[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((X_train_all[feat] < lower) | (X_train_all[feat] > upper)).sum()
    outlier_pct = n_outliers / len(X_train_all) * 100
    if n_outliers > 0:
        outlier_summary[feat] = {'count': n_outliers, 'pct': outlier_pct, 'lower': lower, 'upper': upper}

outlier_df = pd.DataFrame(outlier_summary).T.sort_values('pct', ascending=False)
print("\n=== Outlier Summary (IQR method) ===")
print(outlier_df.head(20))

# Boxplots for features with most outliers
top_outlier_feats = outlier_df.head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

hospital_names = [k for k in splits.keys() if k.startswith('hospital')]
for i, feat in enumerate(top_outlier_feats):
    data_by_client = []
    labels = []
    for name in hospital_names:
        data_by_client.append(splits[name]['X_train'][feat].dropna().values)
        labels.append(name)
    axes[i].boxplot(data_by_client, tick_labels=labels, widths=0.6)
    axes[i].set_title(feat)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/outliers_by_client.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Between-Hospital Distribution Shift (Non-IID Analysis)
# ============================================================
hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}

shift_results = []
for feat in numeric_cols:
    if feat in X_train_all.columns:
        for i, (name1, split1) in enumerate(hospital_splits.items()):
            for name2, split2 in list(hospital_splits.items())[i+1:]:
                stat, pval = stats.ks_2samp(split1['X_train'][feat].dropna(), 
                                             split2['X_train'][feat].dropna())
                shift_results.append({'feature': feat, 'client1': name1, 'client2': name2, 
                                    'ks_stat': stat, 'p_value': pval})

shift_df = pd.DataFrame(shift_results)
significant_shifts = shift_df[shift_df['p_value'] < 0.05]

print(f"\nTotal pairwise comparisons: {len(shift_df)}")
print(f"Significant distribution shifts (p<0.05): {len(significant_shifts)}")
print(f"Features with any significant shift: {significant_shifts['feature'].nunique()}")

feat_shift_summary = significant_shifts.groupby('feature')['ks_stat'].mean().sort_values(ascending=False)
print("\nTop 10 features with largest average KS statistic:")
print(feat_shift_summary.head(10))

# Visualize shift for top features
top_shifted = feat_shift_summary.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(top_shifted):
    for name, split in hospital_splits.items():
        axes[i].hist(split['X_train'][feat].dropna(), bins=30, alpha=0.5, label=name, density=True)
    axes[i].set_title(f'{feat} (KS avg: {feat_shift_summary[feat]:.3f})')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/figures/distribution_shift.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Correlation Analysis
# ============================================================
target_col = y_train_all.columns[0]
correlations = X_train_all.corrwith(y_train_all[target_col]).sort_values(key=abs, ascending=False)

print("\n=== Top 15 Features Correlated with Mortality ===")
print(correlations.head(15))

top_corr_feats = correlations.head(12).index.tolist()
corr_matrix = X_train_all[top_corr_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Matrix: Top Features vs Mortality')
plt.tight_layout()
plt.savefig('results/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Summary Statistics Table
# ============================================================
summary_rows = []
for name, split in splits.items():
    if name.startswith('hospital') or name == 'global':
        X_tr = split['X_train']
        y_tr = split['y_train'].iloc[:,0]
        X_te = split['X_test']
        y_te = split['y_test'].iloc[:,0]
        
        summary_rows.append({
            'Client': name,
            'Train Samples': len(X_tr),
            'Test Samples': len(X_te),
            'Train Pos Rate': f"{y_tr.mean():.3f}",
            'Test Pos Rate': f"{y_te.mean():.3f}",
            'Features': X_tr.shape[1],
            'Missing Train': X_tr.isnull().sum().sum(),
            'Missing Test': X_te.isnull().sum().sum(),
        })

summary_df = pd.DataFrame(summary_rows)
print("\n=== Data Summary Table ===")
print(summary_df.to_string(index=False))

summary_df.to_csv('results/eda_summary.csv', index=False)
print("\nSaved to results/eda_summary.csv")

# ============================================================
# 8. Feature Engineering Validation
# ============================================================
engineered_features = [c for c in X_train_all.columns if any(kw in c for kw in ['_missing', '_min', '_max', '_mean', '_std', '_slope', 'comorbidity', 'elixhauser'])]
print(f"\nEngineered features found: {engineered_features}")

missing_indicators = [c for c in X_train_all.columns if c.endswith('_missing')]
print(f"Missingness indicators: {missing_indicators}")

# ============================================================
# 9. Preprocessing Validation
# ============================================================
print("\n=== Preprocessing Validation ===")

# Verify no data leakage: check that train/test have same columns
for name, split in splits.items():
    train_cols = set(split['X_train'].columns)
    test_cols = set(split['X_test'].columns)
    if train_cols != test_cols:
        print(f"WARNING: {name} - column mismatch!")
        print(f"  Train only: {train_cols - test_cols}")
        print(f"  Test only: {test_cols - train_cols}")
    else:
        print(f"OK: {name} - columns match")

# Verify consistent column order across clients
first_client = list(hospital_splits.keys())[0]
ref_cols = list(hospital_splits[first_client]['X_train'].columns)
for name, split in hospital_splits.items():
    if list(split['X_train'].columns) != ref_cols:
        print(f"WARNING: {name} - column order differs!")
    else:
        print(f"OK: {name} - column order consistent")

print("\nPreprocessing pipeline steps:")
for name, step in pipeline.named_steps.items():
    print(f"  {name}: {type(step).__name__}")

print("\n=== EDA Complete ===")
print("Figures saved to results/figures/")
print("Summary saved to results/eda_summary.csv")