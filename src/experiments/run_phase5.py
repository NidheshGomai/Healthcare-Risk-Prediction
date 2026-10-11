#!/usr/bin/env python
"""
Phase 5 experiment runner -- Federated DT/RF, Naive Bayes, Privacy.
Run: python -m src.experiments.run_phase5 --config configs/phase5.yaml
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
from typing import Dict, List, Tuple
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seeds import set_global_seed, GLOBAL_SEED
from src.federated.federated_forest import run_federated_forest, run_centralized_forest
from src.federated.federated_nb import run_federated_nb, run_centralized_nb
from src.federated.manual_fl import (
    run_federated_learning,
    run_centralized_baseline,
    run_local_only,
)
from src.privacy.secure_agg import SecureAggregation, run_secure_aggregation_demo
from src.privacy.dp import compute_privacy_cost


# ==============================================================
# CLI
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Run Phase 5 experiments')
    parser.add_argument('--config', type=str, default='configs/phase5.yaml')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument(
        '--phase', type=str,
        choices=['forest', 'nb', 'dp', 'secagg', 'all'],
        default='all',
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
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
# Phase 5A -- Federated Forest
# ==============================================================

def run_forest_phase(splits: Dict, cfg: dict, output_dir: Path) -> pd.DataFrame:
    """Run federated forest experiments and return summary table."""
    print("\n" + "=" * 60)
    print("PHASE 5A: FEDERATED DECISION TREES / RANDOM FOREST")
    print("=" * 60)

    forest_cfg = cfg.get('forest', {})
    random_state = cfg.get('random_state', GLOBAL_SEED)
    rows: List[Dict] = []

    for model_type in forest_cfg.get('model_types', ['rf', 'dt']):
        # Centralized baseline
        print(f"\n--- Centralized {model_type.upper()} ---")
        n_est = forest_cfg.get('n_estimators', 50) if model_type == 'rf' else 1
        cent = run_centralized_forest(
            splits, model_type=model_type,
            n_estimators=n_est * 5 if model_type == 'rf' else 1,
            max_depth=forest_cfg.get('max_depth', 10),
            random_state=random_state,
        )
        rows.append({
            'model': model_type, 'setting': 'centralized', 'weighting': '-',
            **{k: v for k, v in cent.items() if k != 'model'},
        })
        print(f"  acc={cent['accuracy']:.4f}, auroc={cent['auroc']:.4f}")

        # Federated with different weightings
        for weighting in forest_cfg.get('weightings', ['uniform']):
            res = run_federated_forest(
                splits, model_type=model_type,
                n_estimators=forest_cfg.get('n_estimators', 50),
                max_depth=forest_cfg.get('max_depth', 10),
                weighting=weighting,
                random_state=random_state,
                output_dir=str(output_dir),
            )
            rows.append({
                'model': model_type,
                'setting': f'federated_{weighting}',
                'weighting': weighting,
                'n_trees': res['n_total_trees'],
                **res['global_metrics'],
            })

            # Local-only averages
            local_accs = [m['accuracy'] for m in res['client_local_metrics'].values()]
            local_aurocs = [m['auroc'] for m in res['client_local_metrics'].values()]
            local_auprcs = [m['auprc'] for m in res['client_local_metrics'].values()]

        # Add local-only row (once per model_type)
        rows.append({
            'model': model_type,
            'setting': 'local_only_avg',
            'weighting': '-',
            'accuracy': float(np.mean(local_accs)),
            'auroc': float(np.mean(local_aurocs)),
            'auprc': float(np.mean(local_auprcs)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'phase5_forest_results.csv', index=False)
    print(f"\n  Forest results saved -> {output_dir / 'phase5_forest_results.csv'}")
    return df


# ==============================================================
# Phase 5B -- Federated Naive Bayes
# ==============================================================

def run_nb_phase(splits: Dict, cfg: dict, output_dir: Path) -> pd.DataFrame:
    """Run federated NB experiments and return summary table."""
    print("\n" + "=" * 60)
    print("PHASE 5B: FEDERATED NAIVE BAYES")
    print("=" * 60)

    nb_cfg = cfg.get('naive_bayes', {})
    random_state = cfg.get('random_state', GLOBAL_SEED)
    rows: List[Dict] = []

    for nb_type in nb_cfg.get('types', ['gaussian', 'categorical']):
        # Centralized baseline
        print(f"\n--- Centralized {nb_type.capitalize()} NB ---")
        cent = run_centralized_nb(splits, nb_type=nb_type)
        rows.append({
            'nb_type': nb_type, 'setting': 'centralized', 'epsilon': None,
            **{k: v for k, v in cent.items() if k != 'model'},
        })
        print(f"  acc={cent['accuracy']:.4f}, auroc={cent['auroc']:.4f}")

        # Federated (no DP) -> should match centralized
        res = run_federated_nb(
            splits, nb_type=nb_type, epsilon=None,
            clip_bound=nb_cfg.get('clip_bound', 10.0),
            random_state=random_state,
            output_dir=str(output_dir),
        )
        rows.append({
            'nb_type': nb_type, 'setting': 'federated', 'epsilon': None,
            **res['global_metrics'],
        })
        if res['verification']:
            rows[-1]['pred_agreement'] = res['verification']['prediction_agreement']
            rows[-1]['prob_rmse'] = res['verification']['probability_rmse']

        # Local-only averages
        local_accs = [m['accuracy'] for m in res['client_local_metrics'].values()]
        local_aurocs = [m['auroc'] for m in res['client_local_metrics'].values()]
        rows.append({
            'nb_type': nb_type, 'setting': 'local_only_avg', 'epsilon': None,
            'accuracy': float(np.mean(local_accs)),
            'auroc': float(np.mean(local_aurocs)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'phase5_nb_results.csv', index=False)
    print(f"\n  NB results saved -> {output_dir / 'phase5_nb_results.csv'}")
    return df


# ==============================================================
# Phase 5C -- Differential Privacy sweep
# ==============================================================

def run_dp_phase(splits: Dict, cfg: dict, output_dir: Path) -> pd.DataFrame:
    """Run DP privacy-utility sweep for NB models."""
    print("\n" + "=" * 60)
    print("PHASE 5C: DIFFERENTIAL PRIVACY -- PRIVACY-UTILITY TRADEOFF")
    print("=" * 60)

    dp_cfg = cfg.get('dp', {})
    nb_cfg = cfg.get('naive_bayes', {})
    random_state = cfg.get('random_state', GLOBAL_SEED)
    epsilons = dp_cfg.get('epsilons', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    mechanism = dp_cfg.get('mechanism', 'laplace')
    n_trials = dp_cfg.get('n_trials', 3)
    rows: List[Dict] = []

    for nb_type in nb_cfg.get('types', ['gaussian', 'categorical']):
        # Centralized (no DP) reference
        cent = run_centralized_nb(splits, nb_type=nb_type)

        for eps in epsilons:
            trial_metrics: List[Dict] = []
            for trial in range(n_trials):
                res = run_federated_nb(
                    splits, nb_type=nb_type, epsilon=eps,
                    dp_mechanism=mechanism,
                    clip_bound=nb_cfg.get('clip_bound', 10.0),
                    random_state=random_state + trial,
                    output_dir=str(output_dir),
                )
                trial_metrics.append(res['global_metrics'])

            # Average over trials
            avg_acc = np.mean([m['accuracy'] for m in trial_metrics])
            std_acc = np.std([m['accuracy'] for m in trial_metrics])
            avg_auroc = np.mean([m['auroc'] for m in trial_metrics])
            std_auroc = np.std([m['auroc'] for m in trial_metrics])
            avg_auprc = np.mean([m['auprc'] for m in trial_metrics])
            std_auprc = np.std([m['auprc'] for m in trial_metrics])

            rows.append({
                'nb_type': nb_type,
                'epsilon': eps,
                'mechanism': mechanism,
                'n_trials': n_trials,
                'accuracy_mean': avg_acc, 'accuracy_std': std_acc,
                'auroc_mean': avg_auroc, 'auroc_std': std_auroc,
                'auprc_mean': avg_auprc, 'auprc_std': std_auprc,
                'centralized_auroc': cent['auroc'],
                'utility_gap': cent['auroc'] - avg_auroc,
            })

            print(f"  {nb_type} eps={eps:.1f}: "
                  f"auroc={avg_auroc:.4f}+/-{std_auroc:.4f}, "
                  f"gap={cent['auroc'] - avg_auroc:+.4f}")

        # Add eps=inf (no DP) row
        res_inf = run_federated_nb(
            splits, nb_type=nb_type, epsilon=None,
            clip_bound=nb_cfg.get('clip_bound', 10.0),
            random_state=random_state,
            output_dir=str(output_dir),
        )
        rows.append({
            'nb_type': nb_type,
            'epsilon': float('inf'),
            'mechanism': 'none',
            'n_trials': 1,
            'accuracy_mean': res_inf['global_metrics']['accuracy'],
            'accuracy_std': 0.0,
            'auroc_mean': res_inf['global_metrics']['auroc'],
            'auroc_std': 0.0,
            'auprc_mean': res_inf['global_metrics']['auprc'],
            'auprc_std': 0.0,
            'centralized_auroc': cent['auroc'],
            'utility_gap': cent['auroc'] - res_inf['global_metrics']['auroc'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'phase5_dp_sweep.csv', index=False)
    print(f"\n  DP sweep saved -> {output_dir / 'phase5_dp_sweep.csv'}")
    return df


# ==============================================================
# Phase 5D -- Secure Aggregation
# ==============================================================

def run_secagg_phase(splits: Dict, cfg: dict, output_dir: Path) -> Dict:
    """Demonstrate secure aggregation on FedAvg logistic."""
    print("\n" + "=" * 60)
    print("PHASE 5D: SECURE AGGREGATION")
    print("=" * 60)

    sa_cfg = cfg.get('secure_agg', {})
    random_state = cfg.get('random_state', GLOBAL_SEED)

    hospital_ids = sorted([k for k in splits if k.startswith('hospital')])

    # 1. Verify mask cancellation
    sa = SecureAggregation(hospital_ids, random_state=random_state)
    cancels = sa.verify_cancellation(shape=(50,))
    print(f"  Mask cancellation verified: {cancels}")

    # 2. Run FedAvg WITHOUT secure agg
    print("\n  --- FedAvg (no SecAgg) ---")
    res_plain = run_federated_learning(
        splits, model_type='logistic',
        num_rounds=sa_cfg.get('num_rounds', 10),
        num_clients=len(hospital_ids),
        use_global_scaler=False,
        output_dir=str(output_dir),
    )
    plain_acc = res_plain['round_metrics'][-1]['weighted_accuracy']
    plain_auroc = res_plain['round_metrics'][-1].get('weighted_auroc', None)

    # 3. Demonstrate SecAgg on the final-round parameters
    # Collect final per-client parameters
    final_params = res_plain['final_global_params']
    client_updates = {}
    for cid in hospital_ids:
        client = res_plain['clients'][cid]
        client_updates[cid] = client.get_parameters()

    demo = run_secure_aggregation_demo(
        client_updates, random_state=random_state,
    )
    print(f"\n  SecAgg demo:")
    print(f"    Masks cancel:      {demo['masks_cancel']}")
    print(f"    Aggregates match:  {demo['aggregates_match']}")
    print(f"    Max param diff:    {demo['max_diff']:.2e}")

    results = {
        'mask_cancellation': cancels,
        'plain_accuracy': plain_acc,
        'plain_auroc': plain_auroc,
        'secagg_masks_cancel': demo['masks_cancel'],
        'secagg_aggregates_match': demo['aggregates_match'],
        'secagg_max_diff': demo['max_diff'],
    }

    summary_df = pd.DataFrame([results])
    summary_df.to_csv(output_dir / 'phase5_secagg_results.csv', index=False)
    print(f"\n  SecAgg results saved -> {output_dir / 'phase5_secagg_results.csv'}")

    return results


# ==============================================================
# Plotting
# ==============================================================

def plot_forest_comparison(forest_df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing centralized, local-only, and federated forest."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, model in enumerate(['rf', 'dt']):
        ax = axes[i]
        mdf = forest_df[forest_df['model'] == model].copy()
        if mdf.empty:
            continue

        labels = mdf['setting'].tolist()
        aurocs = mdf['auroc'].tolist()
        colours = ['#2ecc71' if 'centralized' in s
                   else '#e74c3c' if 'local' in s
                   else '#3498db' for s in labels]

        bars = ax.bar(range(len(labels)), aurocs, color=colours, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('AUROC')
        ax.set_title(f'{"Random Forest" if model == "rf" else "Decision Tree"} -- '
                     f'Centralized vs Federated')
        ax.set_ylim(0.45, max(aurocs) + 0.05)
        ax.grid(axis='y', alpha=0.3)

        for bar, v in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase5_forest_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_dp_utility_curve(dp_df: pd.DataFrame, output_dir: Path):
    """Privacy-utility tradeoff: AUROC vs eps for each NB variant."""
    fig, ax = plt.subplots(figsize=(9, 5))

    markers = {'gaussian': 'o', 'categorical': 's'}
    colours = {'gaussian': '#e74c3c', 'categorical': '#3498db'}

    for nb_type in dp_df['nb_type'].unique():
        sub = dp_df[dp_df['nb_type'] == nb_type].copy()
        # Replace inf with a large number for plotting
        sub.loc[sub['epsilon'] == float('inf'), 'epsilon'] = 100.0

        sub = sub.sort_values('epsilon')
        ax.errorbar(
            sub['epsilon'], sub['auroc_mean'], yerr=sub['auroc_std'],
            marker=markers.get(nb_type, 'o'), label=f'Federated {nb_type.capitalize()} NB',
            color=colours.get(nb_type, 'gray'), capsize=3, linewidth=1.5,
        )
        # Centralized reference line
        cent_auroc = sub['centralized_auroc'].iloc[0]
        ax.axhline(cent_auroc, color=colours.get(nb_type, 'gray'),
                   linestyle='--', alpha=0.5,
                   label=f'Centralized {nb_type.capitalize()} NB')

    ax.set_xscale('log')
    ax.set_xlabel('Privacy Budget eps (log scale)')
    ax.set_ylabel('AUROC')
    ax.set_title('Privacy-Utility Tradeoff (Federated NB with Differential Privacy)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase5_dp_utility_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_nb_verification(nb_df: pd.DataFrame, output_dir: Path):
    """Show that federated NB matches centralized NB."""
    fig, ax = plt.subplots(figsize=(7, 5))

    fed_rows = nb_df[nb_df['setting'] == 'federated']
    cent_rows = nb_df[nb_df['setting'] == 'centralized']

    for nb_type in fed_rows['nb_type'].unique():
        fed = fed_rows[fed_rows['nb_type'] == nb_type]
        cent = cent_rows[cent_rows['nb_type'] == nb_type]
        if fed.empty or cent.empty:
            continue

        fed_auroc = fed['auroc'].values[0]
        cent_auroc = cent['auroc'].values[0]
        pred_agree = fed.get('pred_agreement', pd.Series([None])).values[0]

        ax.scatter(cent_auroc, fed_auroc, s=120, zorder=5,
                   label=f'{nb_type.capitalize()} NB (agree={pred_agree})')

    lims = [0.45, 0.75]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='y = x (perfect match)')
    ax.set_xlabel('Centralized AUROC')
    ax.set_ylabel('Federated AUROC')
    ax.set_title('Federated NB vs Centralized NB -- Equivalence Check')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase5_nb_verification.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase5_summary(
    forest_df: pd.DataFrame,
    nb_df: pd.DataFrame,
    output_dir: Path,
):
    """Comprehensive Phase 5 summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Forest comparison (RF only)
    ax = axes[0]
    rf = forest_df[forest_df['model'] == 'rf']
    if not rf.empty:
        labels = rf['setting'].tolist()
        vals = rf['auroc'].tolist()
        colours = ['#2ecc71' if 'cent' in s
                   else '#e74c3c' if 'local' in s
                   else '#3498db' for s in labels]
        ax.barh(range(len(labels)), vals, color=colours, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('AUROC')
        ax.set_title('Random Forest')
        ax.set_xlim(0.45, max(vals) + 0.05)
        ax.grid(axis='x', alpha=0.3)

    # Panel 2: NB comparison
    ax = axes[1]
    nb_compare = nb_df[nb_df['epsilon'].isna()]
    if not nb_compare.empty:
        labels = [f"{r['nb_type']}_{r['setting']}" for _, r in nb_compare.iterrows()]
        vals = nb_compare['auroc'].tolist()
        colours = ['#2ecc71' if 'cent' in s
                   else '#e74c3c' if 'local' in s
                   else '#9b59b6' for s in labels]
        ax.barh(range(len(labels)), vals, color=colours, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('AUROC')
        ax.set_title('Naive Bayes')
        ax.set_xlim(0.45, max(vals) + 0.05)
        ax.grid(axis='x', alpha=0.3)

    # Panel 3: Privacy-utility (text summary)
    ax = axes[2]
    ax.axis('off')
    summary_text = "Phase 5 Summary\n" + "=" * 30
    if not forest_df.empty:
        best_fed = forest_df[forest_df['setting'].str.contains('federated')]
        if not best_fed.empty:
            best = best_fed.loc[best_fed['auroc'].idxmax()]
            summary_text += f"\n\nBest Fed Forest: {best['setting']}"
            summary_text += f"\n  AUROC: {best['auroc']:.4f}"
    if not nb_df.empty:
        fed_nb = nb_df[(nb_df['setting'] == 'federated') & nb_df['epsilon'].isna()]
        for _, r in fed_nb.iterrows():
            summary_text += f"\n\nFed {r['nb_type'].capitalize()} NB:"
            summary_text += f"\n  AUROC: {r['auroc']:.4f}"
            if 'pred_agreement' in r and pd.notna(r.get('pred_agreement')):
                summary_text += f"\n  Matches centralized: {r['pred_agreement']:.2%}"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'phase5_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================
# Main
# ==============================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get('random_state', GLOBAL_SEED))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed data from {args.data_dir} ...")
    splits = load_processed_splits(args.data_dir)
    print(f"  Splits: {list(splits.keys())}")
    for name, sp in splits.items():
        if 'X_train' in sp:
            print(f"  {name}: train={len(sp['X_train'])}, test={len(sp['X_test'])}")

    forest_df = pd.DataFrame()
    nb_df = pd.DataFrame()
    dp_df = pd.DataFrame()

    if args.phase in ('forest', 'all'):
        forest_df = run_forest_phase(splits, cfg, output_dir)

    if args.phase in ('nb', 'all'):
        nb_df = run_nb_phase(splits, cfg, output_dir)

    if args.phase in ('dp', 'all'):
        dp_df = run_dp_phase(splits, cfg, output_dir)

    if args.phase in ('secagg', 'all'):
        run_secagg_phase(splits, cfg, output_dir)

    # ---- Generate plots ----
    print("\nGenerating Phase 5 plots ...")
    if not forest_df.empty:
        plot_forest_comparison(forest_df, output_dir)
    if not nb_df.empty:
        plot_nb_verification(nb_df, output_dir)
    if not dp_df.empty:
        plot_dp_utility_curve(dp_df, output_dir)
    if not forest_df.empty or not nb_df.empty:
        plot_phase5_summary(forest_df, nb_df, output_dir)

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)

    # ---- Print summary table ----
    if not forest_df.empty:
        print("\n=== Forest Results ===")
        print(forest_df[['model', 'setting', 'auroc']].to_string(index=False))

    if not nb_df.empty:
        print("\n=== NB Results ===")
        cols = ['nb_type', 'setting', 'auroc']
        if 'pred_agreement' in nb_df.columns:
            cols.append('pred_agreement')
        print(nb_df[cols].to_string(index=False))

    if not dp_df.empty:
        print("\n=== DP Privacy-Utility ===")
        print(dp_df[['nb_type', 'epsilon', 'auroc_mean', 'auroc_std',
                      'utility_gap']].to_string(index=False))


if __name__ == '__main__':
    main()
