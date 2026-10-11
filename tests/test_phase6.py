"""
Unit tests for Phase 6 Explainability and Evaluation components:
- Explainability metrics (Rank correlation, Jaccard, Fidelity, Stability)
- Explainers (TreeExplainerWrapper, LinearExplainerWrapper, RuleExtractor, Aggregator)
- Evaluation utilities (ECE, comprehensive classification metrics)
"""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from src.explain.metrics import (
    rank_correlation,
    top_k_jaccard,
    compute_pairwise_consistency,
    surrogate_fidelity_score,
    explanation_stability_score,
)
from src.explain.explainers import (
    TreeExplainerWrapper,
    LinearExplainerWrapper,
    RuleExtractor,
    FederatedExplanationAggregator,
)
from src.evaluation.evaluator import (
    compute_all_metrics,
    compute_expected_calibration_error,
    get_calibration_data,
)


class TestExplainabilityMetrics:
    def test_rank_correlation_identical_and_reversed(self):
        vec_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        vec_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        res_same = rank_correlation(vec_a, vec_b)
        assert np.isclose(res_same['spearman_rho'], 1.0)
        assert np.isclose(res_same['kendall_tau'], 1.0)

        vec_rev = [5.0, 4.0, 3.0, 2.0, 1.0]
        res_rev = rank_correlation(vec_a, vec_rev)
        assert np.isclose(res_rev['spearman_rho'], -1.0)
        assert np.isclose(res_rev['kendall_tau'], -1.0)

    def test_top_k_jaccard(self):
        # Top 3 in a: indices 2, 3, 4 (values 30, 40, 50)
        a = [10, 20, 30, 40, 50]
        # Top 3 in b: indices 2, 3, 4 (values 3, 4, 5)
        b = [1, 2, 3, 4, 5]
        assert top_k_jaccard(a, b, k=3) == 1.0

        # Top 2 in c: indices 0, 1 (values 50, 40)
        c = [50, 40, 10, 20, 30]
        # Top 2 in a: indices 3, 4 (values 40, 50)
        # Intersection is {0, 1} vs {3, 4} -> empty -> 0.0
        assert top_k_jaccard(a, c, k=2) == 0.0

    def test_compute_pairwise_consistency(self):
        client_imps = {
            'c1': np.array([0.5, 0.3, 0.2]),
            'c2': np.array([0.4, 0.4, 0.2]),
            'c3': np.array([0.6, 0.2, 0.2]),
        }
        consistency = compute_pairwise_consistency(client_imps, k_list=(2,))
        assert 'mean_spearman_rho' in consistency
        assert 'mean_jaccard_top2' in consistency
        assert consistency['mean_spearman_rho'] > 0.5

    def test_surrogate_fidelity_score(self):
        y_blackbox = np.array([1, 0, 1, 1, 0])
        y_surrogate = np.array([1, 0, 1, 1, 0])
        fidelity = surrogate_fidelity_score(y_blackbox, y_surrogate)
        assert np.isclose(fidelity['agreement_rate'], 1.0)
        assert np.isclose(fidelity['cohen_kappa'], 1.0)

    def test_explanation_stability_score(self):
        clean = np.array([0.5, 0.3, 0.2])
        perturbed = [
            np.array([0.48, 0.31, 0.21]),
            np.array([0.52, 0.29, 0.19]),
        ]
        stab = explanation_stability_score(clean, perturbed)
        assert stab['mean_cosine_similarity'] > 0.99
        assert stab['mean_rank_stability'] == 1.0


class TestExplainers:
    def test_tree_explainer_wrapper(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

        expl = TreeExplainerWrapper(dt, feature_names=[f"F{i}" for i in range(5)])
        gini = expl.get_feature_importances()
        assert len(gini) == 5
        assert np.isclose(np.sum(gini), 1.0)

        shap_res = expl.compute_shap_values(X[:20], max_samples=20, random_state=42)
        assert 'mean_abs_shap' in shap_res
        assert len(shap_res['mean_abs_shap']) == 5

    def test_linear_explainer_wrapper(self):
        weights = np.array([1.5, -2.0, 0.5])
        expl = LinearExplainerWrapper(weights)
        imp = expl.get_feature_importances()
        np.testing.assert_allclose(imp, [1.5, 2.0, 0.5])

    def test_rule_extractor_clinical_rules(self):
        X, y = make_classification(n_samples=150, n_features=4, random_state=42)
        features = ['HeartRate', 'BloodPressure', 'Lactate', 'Age']

        extractor = RuleExtractor(max_depth=2, min_samples_leaf=10, feature_names=features, random_state=42)
        extractor.fit(X, y)

        assert extractor.fidelity_metrics_['agreement_rate'] > 0.70
        rules = extractor.extract_rules()
        assert len(rules) > 0
        for r in rules:
            assert 'rule_text' in r
            assert 'mortality_risk' in r
            assert 'support_count' in r
            assert 0.0 <= r['mortality_risk'] <= 1.0
            assert r['support_count'] > 0

    def test_federated_explanation_aggregator(self):
        client_imps = {
            'h1': np.array([0.5, 0.3, 0.2]),
            'h2': np.array([0.4, 0.4, 0.2]),
        }
        client_samples = {'h1': 100, 'h2': 200}

        aggregator = FederatedExplanationAggregator(weighting='sample_count')
        res = aggregator.aggregate(client_imps, client_samples)
        assert 'aggregated_importance' in res
        assert np.isclose(np.sum(res['aggregated_importance']), 1.0)

        # With DP noise
        dp_aggregator = FederatedExplanationAggregator(weighting='sample_count', dp_epsilon=1.0, random_state=42)
        dp_res = dp_aggregator.aggregate(client_imps, client_samples)
        assert 'dp_aggregated_importance' in dp_res
        assert np.isclose(np.sum(dp_res['dp_aggregated_importance']), 1.0)


class TestEvaluator:
    def test_compute_all_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4])

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics['accuracy'] == 0.8
        assert 'auroc' in metrics
        assert 'auprc' in metrics
        assert 'brier' in metrics
        assert 'ece' in metrics

    def test_calibration_data(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        prob_true, prob_pred = get_calibration_data(y_true, y_prob, n_bins=2)
        assert len(prob_true) == len(prob_pred)
        assert len(prob_true) > 0
