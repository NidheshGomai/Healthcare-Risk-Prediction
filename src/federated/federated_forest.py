"""
Federated Decision Trees and Random Forest via Forest Union strategy.

Strategy B (One-Shot Forest Union):
    1. Each client trains a local Random Forest (or single Decision Tree).
    2. The server pools ALL trees from every client into one combined ensemble.
    3. Trees can optionally be weighted by client performance or sample count.
    4. The pooled forest predicts via soft-voting (probability averaging).

This is the simplest federated tree strategy — one communication round,
easy to explain, and provides a strong baseline.  Under heavy non-IID
data skew it may underperform histogram-based approaches, but for
moderate heterogeneity it works surprisingly well.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
import warnings
warnings.filterwarnings('ignore')


# ======================================================================
# Pooled Forest (server-side combined ensemble)
# ======================================================================

class PooledForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Combined forest from multiple clients' trees.
    Performs soft voting (probability averaging) across all pooled trees.
    """

    def __init__(
        self,
        trees: Optional[List] = None,
        tree_weights: Optional[List[float]] = None,
    ):
        self.trees: List = trees or []
        self.tree_weights: Optional[List[float]] = tree_weights
        self.classes_: Optional[np.ndarray] = None

    def add_trees(self, new_trees: List, weights: Optional[List[float]] = None):
        """Add trees from a client to the pool."""
        self.trees.extend(new_trees)
        if weights is not None:
            if self.tree_weights is None:
                self.tree_weights = list(weights)
            else:
                self.tree_weights.extend(weights)

    def fit(self, X=None, y=None):
        """No-op — trees are already trained on client data."""
        if self.trees:
            self.classes_ = self.trees[0].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across all trees (soft vote)."""
        if not self.trees:
            raise ValueError("No trees in pool")
        if self.classes_ is None:
            self.classes_ = self.trees[0].classes_

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        avg_proba = np.zeros((n_samples, n_classes))

        if self.tree_weights is not None:
            total_w = sum(self.tree_weights)
            for tree, w in zip(self.trees, self.tree_weights):
                avg_proba += (w / total_w) * tree.predict_proba(X)
        else:
            for tree in self.trees:
                avg_proba += tree.predict_proba(X)
            avg_proba /= len(self.trees)

        return avg_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via soft voting."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def n_trees(self) -> int:
        return len(self.trees)


# ======================================================================
# Client
# ======================================================================

class FederatedForestClient:
    """Local client that trains a Decision Tree or Random Forest."""

    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = 'rf',        # 'rf' or 'dt'
        n_estimators: int = 50,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.train_metrics: Optional[Dict] = None

    def train(self) -> Dict[str, float]:
        """Train the local model and evaluate on the client test set."""
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=self.random_state,
            )

        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        self.train_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'auroc': roc_auc_score(self.y_test, y_prob),
            'auprc': average_precision_score(self.y_test, y_prob),
            'num_examples': len(self.X_train),
        }
        return self.train_metrics

    def get_trees(self) -> List:
        """Return deep-copied tree estimators."""
        if self.model_type == 'rf':
            return [copy.deepcopy(t) for t in self.model.estimators_]
        return [copy.deepcopy(self.model)]

    def get_tree_weights(self, weighting: str = 'uniform') -> List[float]:
        """Compute per-tree weights under the chosen strategy."""
        n = len(self.get_trees()) if self.model else 1
        if weighting == 'uniform':
            return [1.0] * n
        elif weighting == 'performance':
            w = self.train_metrics.get('auroc', 0.5) if self.train_metrics else 0.5
            return [w] * n
        elif weighting == 'sample_count':
            return [float(len(self.X_train))] * n
        return [1.0] * n

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances from the local model."""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


# ======================================================================
# Server
# ======================================================================

class ForestUnionServer:
    """Server that pools trees from all clients."""

    def __init__(self):
        self.pooled_forest: Optional[PooledForestClassifier] = None
        self.client_contributions: Dict[str, int] = {}

    def aggregate_forests(
        self,
        client_trees: Dict[str, Tuple[List, List[float], int]],
        weighting: str = 'uniform',
    ) -> PooledForestClassifier:
        """
        Pool trees from every client into a single ensemble.

        Args:
            client_trees: client_id → (trees, tree_weights, n_train_samples)
            weighting: 'uniform', 'performance', or 'sample_count'.

        Returns:
            PooledForestClassifier with all trees.
        """
        all_trees: List = []
        all_weights: List[float] = []

        for cid, (trees, weights, n_samples) in client_trees.items():
            all_trees.extend(trees)
            if weighting == 'sample_count':
                all_weights.extend([float(n_samples)] * len(trees))
            else:
                all_weights.extend(weights)
            self.client_contributions[cid] = len(trees)

        # Normalise weights
        total = sum(all_weights)
        all_weights = [w / total for w in all_weights]

        self.pooled_forest = PooledForestClassifier(
            trees=all_trees, tree_weights=all_weights
        )
        self.pooled_forest.fit()
        return self.pooled_forest

    def aggregate_feature_importances(
        self,
        client_importances: Dict[str, Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """Weighted-average of per-client feature importances."""
        total_n = sum(n for _, n in client_importances.values())
        agg = np.zeros_like(next(iter(client_importances.values()))[0])
        for _, (imp, n) in client_importances.items():
            agg += (n / total_n) * imp
        return agg


# ======================================================================
# End-to-end runner
# ======================================================================

def run_federated_forest(
    splits: Dict,
    model_type: str = 'rf',
    n_estimators: int = 50,
    max_depth: int = 10,
    weighting: str = 'uniform',
    random_state: int = 42,
    output_dir: str = 'results',
) -> Dict:
    """
    Run the full federated forest experiment.

    Steps:
        1. Each hospital trains a local RF (or DT).
        2. Server pools all trees (Forest Union).
        3. Evaluate pooled forest on global test set and per-client test sets.

    Returns:
        Dict with global_metrics, per-client metrics, pooled forest, and
        aggregated feature importances.
    """
    from pathlib import Path

    hospital_splits = {k: v for k, v in splits.items() if k.startswith('hospital')}

    print(f"\n=== Federated Forest ({model_type.upper()}, weighting={weighting}) ===")
    print(f"  Clients: {len(hospital_splits)}, "
          f"Trees/client: {n_estimators if model_type == 'rf' else 1}")

    # ---------- Train each client ----------
    clients: Dict[str, FederatedForestClient] = {}
    client_trees: Dict[str, Tuple[List, List[float], int]] = {}

    for cid, split in hospital_splits.items():
        client = FederatedForestClient(
            client_id=cid,
            X_train=split['X_train'].values,
            y_train=split['y_train'].values.ravel(),
            X_test=split['X_test'].values,
            y_test=split['y_test'].values.ravel(),
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        metrics = client.train()
        clients[cid] = client

        trees = client.get_trees()
        weights = client.get_tree_weights(weighting=weighting)
        client_trees[cid] = (trees, weights, len(split['X_train']))

        print(f"  {cid}: acc={metrics['accuracy']:.4f}, "
              f"auroc={metrics['auroc']:.4f}, trees={len(trees)}")

    # ---------- Server aggregation ----------
    server = ForestUnionServer()
    pooled = server.aggregate_forests(client_trees, weighting=weighting)
    print(f"  Pooled ensemble: {pooled.n_trees} total trees")

    # ---------- Global evaluation ----------
    X_test_g = splits['global']['X_test'].values
    y_test_g = splits['global']['y_test'].values.ravel()

    y_pred = pooled.predict(X_test_g)
    y_prob = pooled.predict_proba(X_test_g)[:, 1]

    global_metrics = {
        'accuracy': accuracy_score(y_test_g, y_pred),
        'auroc': roc_auc_score(y_test_g, y_prob),
        'auprc': average_precision_score(y_test_g, y_prob),
    }
    print(f"  Global test: acc={global_metrics['accuracy']:.4f}, "
          f"auroc={global_metrics['auroc']:.4f}, auprc={global_metrics['auprc']:.4f}")

    # ---------- Per-client evaluation of pooled model ----------
    client_eval_pooled: Dict[str, Dict] = {}
    for cid, cl in clients.items():
        yp = pooled.predict(cl.X_test)
        ypp = pooled.predict_proba(cl.X_test)[:, 1]
        client_eval_pooled[cid] = {
            'accuracy': accuracy_score(cl.y_test, yp),
            'auroc': roc_auc_score(cl.y_test, ypp),
            'auprc': average_precision_score(cl.y_test, ypp),
        }

    # ---------- Feature importances ----------
    ci = {}
    for cid, cl in clients.items():
        imp = cl.get_feature_importances()
        if imp is not None:
            ci[cid] = (imp, len(cl.X_train))
    fed_importance = server.aggregate_feature_importances(ci) if ci else None

    return {
        'pooled_forest': pooled,
        'global_metrics': global_metrics,
        'client_local_metrics': {cid: c.train_metrics for cid, c in clients.items()},
        'client_eval_pooled': client_eval_pooled,
        'feature_importances': fed_importance,
        'n_total_trees': pooled.n_trees,
    }


def run_centralized_forest(
    splits: Dict,
    model_type: str = 'rf',
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
) -> Dict:
    """Centralized baseline for forest comparison."""
    X_train = splits['global']['X_train'].values
    y_train = splits['global']['y_train'].values.ravel()
    X_test = splits['global']['X_test'].values
    y_test = splits['global']['y_test'].values.ravel()

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=5, class_weight='balanced',
            random_state=random_state, n_jobs=-1,
        )
    else:
        model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=10,
            class_weight='balanced', random_state=random_state,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob),
        'model': model,
    }
