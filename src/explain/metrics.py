"""
Explainability metrics for federated models:
- Cross-client rank consistency (Spearman rho, Kendall tau)
- Top-k feature overlap (Jaccard similarity)
- Surrogate model fidelity (agreement, Cohen's kappa, R²)
- Explanation stability under perturbation / privacy noise
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def rank_correlation(
    ranking_a: Union[np.ndarray, List[float]],
    ranking_b: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Compute Spearman rho and Kendall tau rank correlations between two importance vectors.
    """
    a = np.asarray(ranking_a, dtype=float)
    b = np.asarray(ranking_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Ranking vectors must have identical length.")

    if np.all(a == a[0]) or np.all(b == b[0]):
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0, 'kendall_tau': 0.0, 'kendall_pvalue': 1.0}

    rho, p_rho = spearmanr(a, b)
    tau, p_tau = kendalltau(a, b)

    return {
        'spearman_rho': float(0.0 if np.isnan(rho) else rho),
        'spearman_pvalue': float(1.0 if np.isnan(p_rho) else p_rho),
        'kendall_tau': float(0.0 if np.isnan(tau) else tau),
        'kendall_pvalue': float(1.0 if np.isnan(p_tau) else p_tau),
    }


def top_k_jaccard(
    importance_a: Union[np.ndarray, List[float]],
    importance_b: Union[np.ndarray, List[float]],
    k: int = 5,
) -> float:
    """
    Compute Jaccard similarity of the top-k most important features between two vectors.
    """
    a = np.asarray(importance_a, dtype=float)
    b = np.asarray(importance_b, dtype=float)

    k = min(k, len(a))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])

    intersection = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))
    return float(intersection / union) if union > 0 else 0.0


def compute_pairwise_consistency(
    client_importances: Dict[str, np.ndarray],
    k_list: Tuple[int, ...] = (5, 10),
) -> Dict[str, Union[float, Dict[str, Dict[str, float]]]]:
    """
    Compute mean pairwise Spearman rho, Kendall tau, and Top-k Jaccard across all clients.
    """
    cids = sorted(list(client_importances.keys()))
    n = len(cids)
    if n < 2:
        return {'mean_spearman_rho': 1.0, 'mean_kendall_tau': 1.0, 'mean_jaccard_top5': 1.0}

    spearman_matrix: Dict[str, Dict[str, float]] = {c: {} for c in cids}
    jaccard_top5_matrix: Dict[str, Dict[str, float]] = {c: {} for c in cids}

    spearman_vals: List[float] = []
    kendall_vals: List[float] = []
    jaccard_vals: Dict[int, List[float]] = {k: [] for k in k_list}

    for i in range(n):
        cid_i = cids[i]
        spearman_matrix[cid_i][cid_i] = 1.0
        jaccard_top5_matrix[cid_i][cid_i] = 1.0

        for j in range(i + 1, n):
            cid_j = cids[j]
            corr = rank_correlation(client_importances[cid_i], client_importances[cid_j])
            rho = corr['spearman_rho']
            tau = corr['kendall_tau']

            spearman_vals.append(rho)
            kendall_vals.append(tau)
            spearman_matrix[cid_i][cid_j] = rho
            spearman_matrix[cid_j][cid_i] = rho

            for k in k_list:
                jac = top_k_jaccard(client_importances[cid_i], client_importances[cid_j], k=k)
                jaccard_vals[k].append(jac)
                if k == 5:
                    jaccard_top5_matrix[cid_i][cid_j] = jac
                    jaccard_top5_matrix[cid_j][cid_i] = jac

    results: Dict[str, Any] = {
        'mean_spearman_rho': float(np.mean(spearman_vals)) if spearman_vals else 1.0,
        'std_spearman_rho': float(np.std(spearman_vals)) if spearman_vals else 0.0,
        'mean_kendall_tau': float(np.mean(kendall_vals)) if kendall_vals else 1.0,
        'std_kendall_tau': float(np.std(kendall_vals)) if kendall_vals else 0.0,
        'spearman_matrix': spearman_matrix,
        'jaccard_matrix_top5': jaccard_top5_matrix,
    }

    for k in k_list:
        results[f'mean_jaccard_top{k}'] = float(np.mean(jaccard_vals[k])) if jaccard_vals[k] else 1.0
        results[f'std_jaccard_top{k}'] = float(np.std(jaccard_vals[k])) if jaccard_vals[k] else 0.0

    return results


def surrogate_fidelity_score(
    y_blackbox_preds: np.ndarray,
    y_surrogate_preds: np.ndarray,
) -> Dict[str, float]:
    """
    Measure fidelity: how well a transparent surrogate (e.g. shallow decision tree)
    mimics the complex black-box federated model.
    """
    agreement = float(np.mean(y_blackbox_preds == y_surrogate_preds))
    kappa = float(cohen_kappa_score(y_blackbox_preds, y_surrogate_preds))
    f1 = float(f1_score(y_blackbox_preds, y_surrogate_preds, average='weighted', zero_division=0))

    return {
        'agreement_rate': agreement,
        'cohen_kappa': kappa,
        'surrogate_f1': f1,
    }


def explanation_stability_score(
    clean_importance: np.ndarray,
    perturbed_importances: List[np.ndarray],
) -> Dict[str, float]:
    """
    Assess explanation stability across noise perturbations or random seeds.
    Measures cosine similarity and rank correlation with baseline.
    """
    clean_norm = clean_importance / (np.linalg.norm(clean_importance) + 1e-12)
    cosine_sims = []
    rhos = []

    for pert in perturbed_importances:
        pert_norm = pert / (np.linalg.norm(pert) + 1e-12)
        cos_sim = float(np.dot(clean_norm, pert_norm))
        cosine_sims.append(cos_sim)
        rhos.append(rank_correlation(clean_importance, pert)['spearman_rho'])

    return {
        'mean_cosine_similarity': float(np.mean(cosine_sims)),
        'std_cosine_similarity': float(np.std(cosine_sims)),
        'mean_rank_stability': float(np.mean(rhos)),
        'std_rank_stability': float(np.std(rhos)),
    }
