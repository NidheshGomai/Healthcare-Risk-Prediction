"""
Federated Naive Bayes via sufficient-statistics aggregation.

Key insight
-----------
Naive Bayes parameters are *sufficient statistics* (counts, sums,
sums-of-squares).  Clients release only these aggregates; the server
reconstructs a global NB model that — in the absence of DP noise — is
**mathematically identical** to centralized NB trained on all data.

This also makes it straightforward to add differential-privacy noise:
just perturb the aggregates before sending them.

Supported variants
------------------
- GaussianNB    →  client sends (class_count, Σx, Σx²) per class
- CategoricalNB →  client sends (class_count, per-category histogram) per feature
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import copy
import warnings
warnings.filterwarnings('ignore')


# ======================================================================
# Gaussian NB — Client
# ======================================================================

class FederatedGaussianNBClient:
    """Compute local sufficient statistics for Gaussian NB."""

    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        clip_bound: float = 10.0,
    ):
        self.client_id = client_id
        self.X_train = np.clip(X_train, -clip_bound, clip_bound)
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clip_bound = clip_bound
        self.classes_ = np.unique(y_train)

    def compute_sufficient_stats(self) -> Dict:
        """
        Returns
        -------
        dict with:
            class_count   : (n_classes,)
            feature_sum   : (n_classes, n_features)
            feature_sum_sq: (n_classes, n_features)
            classes       : sorted array of class labels
            n_samples     : total training samples on this client
        """
        n_classes = len(self.classes_)
        n_features = self.X_train.shape[1]

        class_count = np.zeros(n_classes)
        feature_sum = np.zeros((n_classes, n_features))
        feature_sum_sq = np.zeros((n_classes, n_features))

        for c_idx, c in enumerate(self.classes_):
            mask = self.y_train == c
            X_c = self.X_train[mask]
            class_count[c_idx] = mask.sum()
            feature_sum[c_idx] = X_c.sum(axis=0)
            feature_sum_sq[c_idx] = (X_c ** 2).sum(axis=0)

        return {
            'class_count': class_count,
            'feature_sum': feature_sum,
            'feature_sum_sq': feature_sum_sq,
            'classes': self.classes_.copy(),
            'n_samples': len(self.X_train),
        }


# ======================================================================
# Categorical NB — Client
# ======================================================================

class FederatedCategoricalNBClient:
    """Compute local sufficient statistics for Categorical NB."""

    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_categories: Optional[List[int]] = None,
    ):
        self.client_id = client_id
        self.X_train = X_train.astype(int)
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes_ = np.unique(y_train)

        if n_categories is None:
            self.n_categories = [
                int(X_train[:, j].max()) + 1 for j in range(X_train.shape[1])
            ]
        else:
            self.n_categories = list(n_categories)

    def compute_sufficient_stats(self) -> Dict:
        """
        Returns
        -------
        dict with:
            class_count  : (n_classes,)
            feature_count: list of (n_classes, n_categories_j) arrays
            classes      : sorted class labels
            n_categories : list of int
            n_samples    : total training samples on this client
        """
        n_classes = len(self.classes_)
        n_features = self.X_train.shape[1]

        class_count = np.zeros(n_classes)
        feature_count: List[np.ndarray] = []

        for j in range(n_features):
            fc = np.zeros((n_classes, self.n_categories[j]))
            for c_idx, c in enumerate(self.classes_):
                mask = self.y_train == c
                if j == 0:
                    class_count[c_idx] = mask.sum()
                vals = self.X_train[mask, j].astype(int)
                for v in range(self.n_categories[j]):
                    fc[c_idx, v] = (vals == v).sum()
            feature_count.append(fc)

        return {
            'class_count': class_count,
            'feature_count': feature_count,
            'classes': self.classes_.copy(),
            'n_categories': self.n_categories,
            'n_samples': len(self.X_train),
        }


# ======================================================================
# Server
# ======================================================================

class FederatedNBServer:
    """Aggregate NB sufficient statistics from all clients."""

    def __init__(
        self,
        nb_type: str = 'gaussian',
        var_smoothing: float = 1e-9,
        alpha: float = 1.0,
    ):
        self.nb_type = nb_type
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    # ---- Gaussian NB ----

    def aggregate_gaussian(self, client_stats: List[Dict]) -> GaussianNB:
        """
        Sum sufficient statistics across clients and reconstruct a
        GaussianNB model.  Without DP noise, this is mathematically
        identical to fitting centralized GaussianNB on all data.
        """
        total_count = np.zeros_like(client_stats[0]['class_count'])
        total_sum = np.zeros_like(client_stats[0]['feature_sum'])
        total_sumsq = np.zeros_like(client_stats[0]['feature_sum_sq'])

        for st in client_stats:
            total_count += st['class_count']
            total_sum += st['feature_sum']
            total_sumsq += st['feature_sum_sq']

        theta = total_sum / total_count[:, np.newaxis]               # mean
        var = total_sumsq / total_count[:, np.newaxis] - theta ** 2  # variance
        var = np.maximum(var, 1e-9)

        # --- Build sklearn GaussianNB with these parameters ---
        model = GaussianNB(var_smoothing=self.var_smoothing)
        model.classes_ = client_stats[0]['classes']
        model.theta_ = theta
        model.var_ = var + self.var_smoothing * var.max(axis=1, keepdims=True)
        model.class_count_ = total_count
        model.class_prior_ = total_count / total_count.sum()
        model.n_features_in_ = theta.shape[1]

        return model

    # ---- Categorical NB ----

    def aggregate_categorical(self, client_stats: List[Dict]) -> CategoricalNB:
        """
        Sum per-category count histograms across clients and reconstruct
        a CategoricalNB model with Laplace (alpha) smoothing.
        """
        total_count = np.zeros_like(client_stats[0]['class_count'])
        n_features = len(client_stats[0]['feature_count'])
        total_fc = [
            np.zeros_like(client_stats[0]['feature_count'][j])
            for j in range(n_features)
        ]

        for st in client_stats:
            total_count += st['class_count']
            for j in range(n_features):
                total_fc[j] += st['feature_count'][j]

        model = CategoricalNB(alpha=self.alpha)
        model.classes_ = client_stats[0]['classes']
        model.class_count_ = total_count
        model.class_log_prior_ = np.log(total_count / total_count.sum())
        model.n_features_in_ = n_features
        model.category_count_ = total_fc

        model.feature_log_prob_ = []
        for j in range(n_features):
            smoothed = total_fc[j] + self.alpha
            log_prob = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
            model.feature_log_prob_.append(log_prob)

        return model


# ======================================================================
# End-to-end runner
# ======================================================================

def run_federated_nb(
    splits: Dict,
    nb_type: str = 'gaussian',
    epsilon: Optional[float] = None,
    dp_mechanism: str = 'laplace',
    clip_bound: float = 10.0,
    random_state: int = 42,
    output_dir: str = 'results',
) -> Dict:
    """
    Run a full federated NB experiment.

    Args:
        splits: processed data splits (hospital_* + global).
        nb_type: 'gaussian' or 'categorical'.
        epsilon: DP budget per client (None → no DP).
        dp_mechanism: 'laplace' or 'gaussian'.
        clip_bound: feature-value clipping bound for DP.
        random_state: RNG seed.
        output_dir: where to save results.

    Returns:
        Dict with global_metrics, per-client local metrics, verification
        results, and the epsilon used.
    """
    from pathlib import Path

    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}

    dp_str = f" + DP (ε={epsilon})" if epsilon is not None else ""
    print(f"\n=== Federated {nb_type.capitalize()} NB{dp_str} ===")

    # Determine n_categories for categorical NB
    n_categories = None
    if nb_type == 'categorical':
        all_X = np.vstack([s['X_train'].values for s in hospital_splits.values()])
        n_categories = [int(all_X[:, j].max()) + 1 for j in range(all_X.shape[1])]

    # ---- Collect stats from each client ----
    client_stats: List[Dict] = []
    client_local_metrics: Dict[str, Dict] = {}

    for cid, split in hospital_splits.items():
        X_tr = split['X_train'].values
        y_tr = split['y_train'].values.ravel()
        X_te = split['X_test'].values
        y_te = split['y_test'].values.ravel()

        if nb_type == 'gaussian':
            client = FederatedGaussianNBClient(
                cid, X_tr, y_tr, X_te, y_te, clip_bound=clip_bound,
            )
        else:
            client = FederatedCategoricalNBClient(
                cid, X_tr, y_tr, X_te, y_te, n_categories=n_categories,
            )

        stats = client.compute_sufficient_stats()

        # Add DP noise if requested
        if epsilon is not None:
            from src.privacy.dp import (
                add_dp_noise_to_gaussian_stats,
                add_dp_noise_to_categorical_stats,
            )
            if nb_type == 'gaussian':
                stats = add_dp_noise_to_gaussian_stats(
                    stats, epsilon=epsilon, clip_bound=clip_bound,
                    mechanism=dp_mechanism,
                    random_state=random_state + abs(hash(cid)) % 10000,
                )
            else:
                stats = add_dp_noise_to_categorical_stats(
                    stats, epsilon=epsilon, mechanism=dp_mechanism,
                    random_state=random_state + abs(hash(cid)) % 10000,
                )

        client_stats.append(stats)

        # Local model for comparison
        if nb_type == 'gaussian':
            local_m = GaussianNB(var_smoothing=1e-9)
        else:
            local_m = CategoricalNB(alpha=1.0)

        local_m.fit(X_tr, y_tr)
        yp = local_m.predict(X_te)
        ypp = local_m.predict_proba(X_te)[:, 1]
        client_local_metrics[cid] = {
            'accuracy': accuracy_score(y_te, yp),
            'auroc': roc_auc_score(y_te, ypp),
            'auprc': average_precision_score(y_te, ypp),
        }
        print(f"  {cid} (local): acc={client_local_metrics[cid]['accuracy']:.4f}, "
              f"auroc={client_local_metrics[cid]['auroc']:.4f}")

    # ---- Server aggregation ----
    server = FederatedNBServer(
        nb_type=nb_type,
        var_smoothing=1e-9,
        alpha=1.0,
    )
    if nb_type == 'gaussian':
        global_model = server.aggregate_gaussian(client_stats)
    else:
        global_model = server.aggregate_categorical(client_stats)

    # ---- Global test evaluation ----
    X_te_g = splits['global']['X_test'].values
    y_te_g = splits['global']['y_test'].values.ravel()

    y_pred = global_model.predict(X_te_g)
    y_prob = global_model.predict_proba(X_te_g)[:, 1]

    global_metrics = {
        'accuracy': accuracy_score(y_te_g, y_pred),
        'auroc': roc_auc_score(y_te_g, y_prob),
        'auprc': average_precision_score(y_te_g, y_prob),
    }
    print(f"  Federated global: acc={global_metrics['accuracy']:.4f}, "
          f"auroc={global_metrics['auroc']:.4f}, auprc={global_metrics['auprc']:.4f}")

    # ---- Verification against centralized (only when no DP) ----
    verification: Dict = {}
    if epsilon is None:
        if nb_type == 'gaussian':
            central = GaussianNB(var_smoothing=1e-9)
        else:
            central = CategoricalNB(alpha=1.0)

        X_all = np.vstack([s['X_train'].values for s in hospital_splits.values()])
        y_all = np.concatenate([s['y_train'].values.ravel() for s in hospital_splits.values()])
        central.fit(X_all, y_all)

        yp_c = central.predict(X_te_g)
        ypp_c = central.predict_proba(X_te_g)[:, 1]

        central_m = {
            'accuracy': accuracy_score(y_te_g, yp_c),
            'auroc': roc_auc_score(y_te_g, ypp_c),
            'auprc': average_precision_score(y_te_g, ypp_c),
        }
        pred_agree = float(np.mean(y_pred == yp_c))
        prob_rmse = float(np.sqrt(np.mean((y_prob - ypp_c) ** 2)))

        verification = {
            'centralized_metrics': central_m,
            'prediction_agreement': pred_agree,
            'probability_rmse': prob_rmse,
        }
        print(f"  Centralized:     acc={central_m['accuracy']:.4f}, "
              f"auroc={central_m['auroc']:.4f}")
        print(f"  Verification:    pred_agreement={pred_agree:.4f}, "
              f"prob_RMSE={prob_rmse:.6f}")

    return {
        'model': global_model,
        'global_metrics': global_metrics,
        'client_local_metrics': client_local_metrics,
        'verification': verification,
        'epsilon': epsilon,
    }


def run_centralized_nb(
    splits: Dict,
    nb_type: str = 'gaussian',
) -> Dict:
    """Centralized NB baseline for comparison."""
    X_train = splits['global']['X_train'].values
    y_train = splits['global']['y_train'].values.ravel()
    X_test = splits['global']['X_test'].values
    y_test = splits['global']['y_test'].values.ravel()

    if nb_type == 'gaussian':
        model = GaussianNB(var_smoothing=1e-9)
    else:
        model = CategoricalNB(alpha=1.0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob),
        'model': model,
    }
