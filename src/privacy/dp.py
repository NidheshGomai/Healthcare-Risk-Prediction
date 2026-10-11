"""
Differential Privacy mechanisms for federated learning.

Implements:
- Laplace mechanism (pure ε-DP)
- Gaussian mechanism ((ε,δ)-DP)
- DP noise injection for Naive Bayes sufficient statistics
- Privacy budget accountant with basic and advanced composition
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class LaplaceMechanism:
    """
    Laplace mechanism for ε-differential privacy.
    Adds noise ~ Laplace(0, sensitivity/ε) to query results.
    """

    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(
        self, value: Union[float, np.ndarray], random_state: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """Add Laplace noise to a value or array."""
        rng = np.random.RandomState(random_state)
        if isinstance(value, np.ndarray):
            noise = rng.laplace(0, self.scale, value.shape)
        else:
            noise = rng.laplace(0, self.scale)
        return value + noise

    def get_noise_scale(self) -> float:
        return self.scale


class GaussianMechanism:
    """
    Gaussian mechanism for (ε,δ)-differential privacy.
    Adds noise ~ N(0, σ²) where σ = sensitivity × √(2·ln(1.25/δ)) / ε.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def add_noise(
        self, value: Union[float, np.ndarray], random_state: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """Add Gaussian noise to a value or array."""
        rng = np.random.RandomState(random_state)
        if isinstance(value, np.ndarray):
            noise = rng.normal(0, self.sigma, value.shape)
        else:
            noise = rng.normal(0, self.sigma)
        return value + noise

    def get_noise_scale(self) -> float:
        return self.sigma


class PrivacyAccountant:
    """
    Track cumulative privacy budget across multiple queries/rounds.
    Supports basic sequential composition and advanced composition.
    """

    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.queries: List[Tuple[float, float]] = []
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0

    def spend(self, epsilon: float, delta: float = 0.0):
        """Record a privacy expenditure."""
        self.queries.append((epsilon, delta))
        self.spent_epsilon += epsilon
        self.spent_delta += delta

    def remaining_epsilon(self, composition: str = 'basic') -> float:
        """Compute remaining epsilon budget under the chosen composition theorem."""
        if composition == 'basic':
            return max(0, self.total_epsilon - self.spent_epsilon)
        elif composition == 'advanced':
            k = len(self.queries)
            if k == 0:
                return self.total_epsilon
            eps_per_query = self.spent_epsilon / k
            advanced_spent = (
                np.sqrt(2 * k * np.log(1 / self.total_delta)) * eps_per_query
                + k * eps_per_query * (np.exp(eps_per_query) - 1)
            )
            return max(0, self.total_epsilon - advanced_spent)
        return max(0, self.total_epsilon - self.spent_epsilon)

    def can_query(self, epsilon: float) -> bool:
        """Check if we have enough budget for another query."""
        return self.remaining_epsilon() >= epsilon

    @property
    def budget_exhausted(self) -> bool:
        return self.remaining_epsilon() <= 0

    def get_summary(self) -> Dict:
        return {
            'total_epsilon': self.total_epsilon,
            'total_delta': self.total_delta,
            'spent_epsilon': self.spent_epsilon,
            'spent_delta': self.spent_delta,
            'remaining_epsilon_basic': self.remaining_epsilon('basic'),
            'remaining_epsilon_advanced': self.remaining_epsilon('advanced'),
            'n_queries': len(self.queries),
        }


# ---------------------------------------------------------------------------
# DP wrappers for Naive Bayes sufficient statistics
# ---------------------------------------------------------------------------

def add_dp_noise_to_gaussian_stats(
    stats: Dict,
    epsilon: float,
    clip_bound: float = 10.0,
    mechanism: str = 'laplace',
    delta: float = 1e-5,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Add DP noise to Gaussian NB sufficient statistics.

    The epsilon budget is split equally across three queries:
        1. class_count  (sensitivity = 1)
        2. feature_sum  (sensitivity = clip_bound)
        3. feature_sum_sq (sensitivity = clip_bound²)

    Args:
        stats: Dict with class_count, feature_sum, feature_sum_sq arrays.
        epsilon: Total privacy budget for this client's release.
        clip_bound: Per-feature contribution bound.
        mechanism: 'laplace' or 'gaussian'.
        delta: Only used when mechanism='gaussian'.
        random_state: RNG seed.
    """
    eps_per_query = epsilon / 3.0

    if mechanism == 'laplace':
        count_mech = LaplaceMechanism(eps_per_query, sensitivity=1.0)
        sum_mech = LaplaceMechanism(eps_per_query, sensitivity=clip_bound)
        sumsq_mech = LaplaceMechanism(eps_per_query, sensitivity=clip_bound ** 2)
    else:
        delta_per = delta / 3.0
        count_mech = GaussianMechanism(eps_per_query, delta=delta_per, sensitivity=1.0)
        sum_mech = GaussianMechanism(eps_per_query, delta=delta_per, sensitivity=clip_bound)
        sumsq_mech = GaussianMechanism(eps_per_query, delta=delta_per, sensitivity=clip_bound ** 2)

    rng_base = random_state if random_state is not None else None

    noisy_count = np.maximum(
        count_mech.add_noise(stats['class_count'].astype(float), random_state=rng_base),
        1.0,  # at least 1 to avoid division by zero
    )

    noisy_sum = sum_mech.add_noise(
        stats['feature_sum'].astype(float),
        random_state=(rng_base + 1) if rng_base is not None else None,
    )

    noisy_sumsq = sumsq_mech.add_noise(
        stats['feature_sum_sq'].astype(float),
        random_state=(rng_base + 2) if rng_base is not None else None,
    )

    # Ensure variance remains positive: var = sumsq/n - mean²  ≥ 1e-9
    noisy_mean = noisy_sum / noisy_count[:, np.newaxis]
    noisy_sumsq = np.maximum(
        noisy_sumsq,
        noisy_mean ** 2 * noisy_count[:, np.newaxis] + 1e-9,
    )

    return {
        'class_count': noisy_count,
        'feature_sum': noisy_sum,
        'feature_sum_sq': noisy_sumsq,
        'classes': stats['classes'],
        'n_samples': stats['n_samples'],
    }


def add_dp_noise_to_categorical_stats(
    stats: Dict,
    epsilon: float,
    mechanism: str = 'laplace',
    delta: float = 1e-5,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Add DP noise to Categorical NB sufficient statistics.

    The budget is split across 1 + n_features queries:
        - 1 for class_count (sensitivity = 1)
        - n_features for each feature_count histogram (sensitivity = 1 each)

    Args:
        stats: Dict with class_count and feature_count arrays.
        epsilon: Total privacy budget.
        mechanism: 'laplace' or 'gaussian'.
        delta: Only used when mechanism='gaussian'.
        random_state: RNG seed.
    """
    n_features = len(stats['feature_count'])
    eps_per_query = epsilon / (1 + n_features)

    if mechanism == 'laplace':
        count_mech = LaplaceMechanism(eps_per_query, sensitivity=1.0)
        feat_mech = LaplaceMechanism(eps_per_query, sensitivity=1.0)
    else:
        delta_per = delta / (1 + n_features)
        count_mech = GaussianMechanism(eps_per_query, delta=delta_per, sensitivity=1.0)
        feat_mech = GaussianMechanism(eps_per_query, delta=delta_per, sensitivity=1.0)

    rng_base = random_state if random_state is not None else None

    noisy_count = np.maximum(
        count_mech.add_noise(stats['class_count'].astype(float), random_state=rng_base),
        1.0,
    )

    noisy_feature_count = []
    for j, fc in enumerate(stats['feature_count']):
        rs = (rng_base + j + 1) if rng_base is not None else None
        noisy_fc = np.maximum(feat_mech.add_noise(fc.astype(float), random_state=rs), 0.0)
        noisy_feature_count.append(noisy_fc)

    return {
        'class_count': noisy_count,
        'feature_count': noisy_feature_count,
        'classes': stats['classes'],
        'n_categories': stats['n_categories'],
        'n_samples': stats['n_samples'],
    }


# ---------------------------------------------------------------------------
# Privacy cost helpers
# ---------------------------------------------------------------------------

def compute_privacy_cost(
    n_rounds: int,
    epsilon_per_round: float,
    composition: str = 'basic',
    delta: float = 1e-5,
) -> Dict:
    """
    Compute total privacy cost for a federated learning run.

    Args:
        n_rounds: Number of FL rounds.
        epsilon_per_round: Privacy budget consumed per round.
        composition: 'basic' (linear) or 'advanced' (sub-linear).
        delta: Privacy parameter δ.

    Returns:
        Dict with total ε, δ, and per-round budget.
    """
    if composition == 'basic':
        total_epsilon = n_rounds * epsilon_per_round
        total_delta = n_rounds * delta
    elif composition == 'advanced':
        k = n_rounds
        eps = epsilon_per_round
        total_epsilon = (
            np.sqrt(2 * k * np.log(1 / delta)) * eps
            + k * eps * (np.exp(eps) - 1)
        )
        total_delta = delta
    else:
        total_epsilon = n_rounds * epsilon_per_round
        total_delta = n_rounds * delta

    return {
        'total_epsilon': total_epsilon,
        'total_delta': total_delta,
        'n_rounds': n_rounds,
        'epsilon_per_round': epsilon_per_round,
        'composition': composition,
    }
