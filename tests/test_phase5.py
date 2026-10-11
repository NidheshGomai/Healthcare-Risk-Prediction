"""
Unit tests for Phase 5 components:
- Differential Privacy (Laplace, Gaussian, Accountant, noise injection)
- Secure Aggregation (Pairwise Masking, Cancellation, Parameter Aggregation)
- Federated Forest (Pooled Forest Classifier, Forest Union Server)
- Federated Naive Bayes (Sufficient Statistics Aggregation, Centralized Equivalence)
"""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.privacy.dp import (
    LaplaceMechanism,
    GaussianMechanism,
    PrivacyAccountant,
    compute_privacy_cost,
    add_dp_noise_to_gaussian_stats,
    add_dp_noise_to_categorical_stats,
)
from src.privacy.secure_agg import (
    SecureAggregation,
    run_secure_aggregation_demo,
)
from src.federated.federated_forest import (
    PooledForestClassifier,
    FederatedForestClient,
    ForestUnionServer,
)
from src.federated.federated_nb import (
    FederatedGaussianNBClient,
    FederatedNBServer,
)


class TestDifferentialPrivacy:
    def test_laplace_mechanism(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        shape = (100, 10)
        val = np.zeros(shape)
        noisy = mech.add_noise(val, random_state=42)
        assert noisy.shape == shape
        # Laplace(0, 1.0) variance is 2 * (1.0)^2 = 2.0
        assert 1.0 < np.var(noisy) < 3.0

    def test_gaussian_mechanism(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        val = np.zeros((50, 5))
        noisy = mech.add_noise(val, random_state=42)
        assert noisy.shape == (50, 5)
        assert not np.isnan(noisy).any()

    def test_privacy_accountant(self):
        accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-4)
        accountant.spend(epsilon=1.0, delta=1e-5)
        accountant.spend(epsilon=2.0, delta=1e-5)

        assert accountant.spent_epsilon == 3.0
        assert np.isclose(accountant.remaining_epsilon('basic'), 7.0)
        assert accountant.can_query(5.0)
        assert not accountant.budget_exhausted

        summary = accountant.get_summary()
        assert summary['n_queries'] == 2
        assert summary['total_epsilon'] == 10.0

    def test_compute_privacy_cost(self):
        cost_basic = compute_privacy_cost(n_rounds=10, epsilon_per_round=0.5, composition='basic')
        assert cost_basic['total_epsilon'] == 5.0

        cost_adv = compute_privacy_cost(n_rounds=10, epsilon_per_round=0.5, composition='advanced', delta=1e-5)
        assert cost_adv['total_epsilon'] > 0

    def test_dp_noise_gaussian_stats(self):
        stats = {
            'class_count': np.array([50, 50]),
            'feature_sum': np.ones((2, 5)) * 10.0,
            'feature_sum_sq': np.ones((2, 5)) * 50.0,
            'classes': np.array([0, 1]),
            'n_samples': 100,
        }
        noisy = add_dp_noise_to_gaussian_stats(stats, epsilon=1.0, clip_bound=5.0, random_state=42)
        assert noisy['class_count'].shape == stats['class_count'].shape
        assert (noisy['class_count'] >= 1.0).all()
        assert noisy['feature_sum'].shape == stats['feature_sum'].shape
        assert noisy['feature_sum_sq'].shape == stats['feature_sum_sq'].shape


class TestSecureAggregation:
    def test_mask_cancellation(self):
        client_ids = ['hospital_0', 'hospital_1', 'hospital_2']
        sec_agg = SecureAggregation(client_ids=client_ids, random_state=42)
        assert sec_agg.verify_cancellation(shape=(25,))

    def test_aggregate_matches_true_average(self):
        client_ids = ['c1', 'c2', 'c3']
        sec_agg = SecureAggregation(client_ids=client_ids, random_state=42)
        
        # Real client updates (e.g. weights and biases)
        rng = np.random.RandomState(42)
        true_updates = {
            cid: [rng.randn(10, 2), rng.randn(2)]
            for cid in client_ids
        }

        # Mask updates
        masked_updates = {
            cid: sec_agg.mask_update(cid, update)
            for cid, update in true_updates.items()
        }

        # Aggregate masked updates
        agg_masked = sec_agg.aggregate(masked_updates)

        # Compute true average directly
        expected_w = np.mean([true_updates[cid][0] for cid in client_ids], axis=0)
        expected_b = np.mean([true_updates[cid][1] for cid in client_ids], axis=0)

        np.testing.assert_allclose(agg_masked[0], expected_w, atol=1e-10)
        np.testing.assert_allclose(agg_masked[1], expected_b, atol=1e-10)

    def test_secure_agg_demo_helper(self):
        client_updates = {
            'c1': [np.array([1.0, 2.0]), np.array([3.0])],
            'c2': [np.array([3.0, 4.0]), np.array([5.0])],
        }
        demo = run_secure_aggregation_demo(client_updates, random_state=42)
        assert demo['masks_cancel'] is True
        assert demo['aggregates_match'] is True
        assert demo['max_diff'] < 1e-10


class TestFederatedForest:
    def test_pooled_forest_predict_and_proba(self):
        X, y = make_classification(n_samples=150, n_features=5, random_state=42)
        
        t1 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X[:75], y[:75])
        t2 = DecisionTreeClassifier(max_depth=3, random_state=43).fit(X[75:], y[75:])

        ensemble = PooledForestClassifier(trees=[t1, t2])
        ensemble.fit()

        assert ensemble.n_trees == 2
        proba = ensemble.predict_proba(X)
        assert proba.shape == (150, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(150), atol=1e-5)

        preds = ensemble.predict(X)
        assert len(preds) == 150
        assert set(preds).issubset({0, 1})

    def test_forest_union_server(self):
        X, y = make_classification(n_samples=200, n_features=6, random_state=42)

        c1 = FederatedForestClient('c1', X[:100], y[:100], X[100:], y[100:], model_type='rf', n_estimators=5)
        c2 = FederatedForestClient('c2', X[100:], y[100:], X[:100], y[:100], model_type='rf', n_estimators=5)

        c1.train()
        c2.train()

        server = ForestUnionServer()
        client_trees = {
            'c1': (c1.get_trees(), c1.get_tree_weights('uniform'), len(c1.X_train)),
            'c2': (c2.get_trees(), c2.get_tree_weights('uniform'), len(c2.X_train)),
        }
        pooled = server.aggregate_forests(client_trees, weighting='uniform')

        assert pooled.n_trees == 10
        probs = pooled.predict_proba(X)
        assert probs.shape == (200, 2)


class TestFederatedNaiveBayes:
    def test_gaussian_nb_equivalence(self):
        X, y = make_classification(n_samples=300, n_features=6, random_state=42)

        # Centralized GaussianNB
        gnb_central = GaussianNB(var_smoothing=1e-9)
        gnb_central.fit(X, y)

        # 3 clients
        c1 = FederatedGaussianNBClient('c1', X[:100], y[:100], X, y)
        c2 = FederatedGaussianNBClient('c2', X[100:200], y[100:200], X, y)
        c3 = FederatedGaussianNBClient('c3', X[200:], y[200:], X, y)

        stats = [
            c1.compute_sufficient_stats(),
            c2.compute_sufficient_stats(),
            c3.compute_sufficient_stats(),
        ]

        server = FederatedNBServer(nb_type='gaussian', var_smoothing=1e-9)
        fed_model = server.aggregate_gaussian(stats)

        # Compare predicted probabilities
        prob_central = gnb_central.predict_proba(X)
        prob_fed = fed_model.predict_proba(X)

        np.testing.assert_allclose(prob_central, prob_fed, atol=1e-4)
