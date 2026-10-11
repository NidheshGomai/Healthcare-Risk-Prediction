"""
Explainability tools for federated learning:
- TreeExplainerWrapper (TreeSHAP and Impurity Importance)
- LinearExplainerWrapper (Coefficient Importance)
- RuleExtractor (Clinician-readable shallow decision tree surrogate)
- FederatedExplanationAggregator (Privacy-preserving summary aggregation with optional DP)
"""
import copy
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.explain.metrics import surrogate_fidelity_score
from src.privacy.dp import LaplaceMechanism


class TreeExplainerWrapper:
    """
    Computes global and local explanations for Tree and Random Forest models.
    Supports Gini importance, permutation importance, and TreeSHAP.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names

    def get_feature_importances(self) -> np.ndarray:
        """Return Gini impurity-based feature importances."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'trees') and len(self.model.trees) > 0:
            # PooledForestClassifier
            imps = [t.feature_importances_ for t in self.model.trees if hasattr(t, 'feature_importances_')]
            if imps:
                return np.mean(imps, axis=0)
        raise ValueError("Model does not provide feature_importances_.")

    def compute_permutation_importance(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scoring: str = 'roc_auc',
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Compute permutation feature importance on validation data."""
        res = permutation_importance(
            self.model, X_val, y_val, scoring=scoring, n_repeats=n_repeats, random_state=random_state
        )
        return {
            'importance_mean': res.importances_mean,
            'importance_std': res.importances_std,
        }

    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: int = 150,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Compute TreeSHAP values on background/test samples.
        Returns mean absolute SHAP values per feature and raw SHAP matrix.
        """
        if not HAS_SHAP:
            # Fallback if SHAP is unavailable
            gini = self.get_feature_importances()
            return {
                'mean_abs_shap': gini,
                'shap_values': np.tile(gini, (len(X), 1)),
                'base_value': 0.5,
                'has_shap': False,
            }

        rng = np.random.RandomState(random_state)
        if len(X) > max_samples:
            indices = rng.choice(len(X), size=max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # For PooledForestClassifier, pick a representative sub-forest or individual tree
        model_to_explain = self.model
        if hasattr(self.model, 'trees') and len(self.model.trees) > 0:
            # Explain a subsample of trees if pooled
            model_to_explain = self.model.trees[0]

        try:
            explainer = shap.TreeExplainer(model_to_explain)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification shapes (list of [class0, class1] or (n, m, 2))
            if isinstance(shap_values, list) and len(shap_values) == 2:
                vals = shap_values[1]  # positive class (mortality)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = np.array(shap_values)

            mean_abs = np.mean(np.abs(vals), axis=0)
            return {
                'mean_abs_shap': mean_abs,
                'shap_values': vals,
                'X_sample': X_sample,
                'base_value': float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
                'has_shap': True,
            }
        except Exception:
            # Fallback to feature importance
            gini = self.get_feature_importances()
            return {
                'mean_abs_shap': gini,
                'shap_values': np.tile(gini, (len(X_sample), 1)),
                'base_value': 0.5,
                'has_shap': False,
            }


class LinearExplainerWrapper:
    """Computes feature importance for linear / logistic models."""

    def __init__(self, weights: np.ndarray, feature_names: Optional[List[str]] = None):
        self.weights = weights.flatten()
        self.feature_names = feature_names

    def get_feature_importances(self) -> np.ndarray:
        """Magnitude of standardized coefficients."""
        return np.abs(self.weights)


class RuleExtractor:
    """
    Trains a shallow, transparent surrogate decision tree on the complex federated model's predictions
    to produce clinician-interpretable IF-THEN clinical decision rules.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_names = feature_names
        self.random_state = random_state
        self.surrogate_tree: Optional[DecisionTreeClassifier] = None
        self.fidelity_metrics_: Optional[Dict[str, float]] = None

    def fit(self, X: np.ndarray, y_blackbox_preds: np.ndarray) -> "RuleExtractor":
        """Fit surrogate tree to black-box predictions."""
        self.surrogate_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.surrogate_tree.fit(X, y_blackbox_preds)

        y_surr_preds = self.surrogate_tree.predict(X)
        self.fidelity_metrics_ = surrogate_fidelity_score(y_blackbox_preds, y_surr_preds)
        return self

    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        Traverse the surrogate tree and generate IF-THEN rules with clinical conditions,
        support, predicted risk, and leaf sample counts.
        """
        if self.surrogate_tree is None:
            raise ValueError("Surrogate tree has not been fitted.")

        tree = self.surrogate_tree.tree_
        feature_names = self.feature_names or [f"Feature_{i}" for i in range(tree.n_features)]
        rules: List[Dict[str, Any]] = []

        def recurse(node: int, current_rule: List[str]):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left branch: feature <= threshold
                recurse(tree.children_left[node], current_rule + [f"{name} <= {threshold:.2f}"])
                # Right branch: feature > threshold
                recurse(tree.children_right[node], current_rule + [f"{name} > {threshold:.2f}"])
            else:
                # Leaf node
                counts = tree.value[node][0]
                total = counts.sum()
                prob = counts[1] / total if len(counts) > 1 and total > 0 else 0.0
                pred_class = int(np.argmax(counts))

                condition_str = " AND ".join(current_rule) if current_rule else "TRUE (All patients)"
                rule_text = (
                    f"IF {condition_str} THEN Predicted Mortality = {pred_class} "
                    f"(Risk: {prob*100:.1f}%, Support: {int(total)} patients)"
                )

                rules.append({
                    'condition': condition_str,
                    'conditions_list': list(current_rule),
                    'predicted_class': pred_class,
                    'mortality_risk': float(prob),
                    'support_count': int(total),
                    'rule_text': rule_text,
                })

        recurse(0, [])
        return sorted(rules, key=lambda r: r['support_count'], reverse=True)


class FederatedExplanationAggregator:
    """
    Aggregates per-client local explanations (feature importance / mean |SHAP|) at the server,
    with optional Differential Privacy noise injection.
    """

    def __init__(
        self,
        weighting: str = 'sample_count',
        dp_epsilon: Optional[float] = None,
        random_state: int = 42,
    ):
        self.weighting = weighting
        self.dp_epsilon = dp_epsilon
        self.random_state = random_state

    def aggregate(
        self,
        client_importances: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate local feature explanations.
        """
        cids = list(client_importances.keys())
        first_imp = client_importances[cids[0]]
        n_features = len(first_imp)

        if self.weighting == 'sample_count' and client_sample_counts is not None:
            total_n = sum(client_sample_counts[c] for c in cids)
            weights = {c: client_sample_counts[c] / total_n for c in cids}
        else:
            weights = {c: 1.0 / len(cids) for c in cids}

        # Weighted sum
        aggregated = np.zeros(n_features, dtype=float)
        for c in cids:
            # Normalize local importance to sum to 1 before aggregating
            imp = client_importances[c]
            norm_imp = imp / (np.sum(imp) + 1e-12)
            aggregated += weights[c] * norm_imp

        # Final normalization
        aggregated = aggregated / (np.sum(aggregated) + 1e-12)

        result = {
            'aggregated_importance': aggregated,
        }

        # Optional Differential Privacy
        if self.dp_epsilon is not None and self.dp_epsilon > 0:
            # Sensitivity of normalized importance bounded by 1.0 / n_clients
            sensitivity = 1.0 / len(cids)
            mech = LaplaceMechanism(epsilon=self.dp_epsilon, sensitivity=sensitivity)
            noisy_imp = mech.add_noise(aggregated, random_state=self.random_state)
            noisy_imp = np.maximum(noisy_imp, 0.0)
            noisy_imp = noisy_imp / (np.sum(noisy_imp) + 1e-12)
            result['dp_aggregated_importance'] = noisy_imp

        return result
