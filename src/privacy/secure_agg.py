"""
Secure Aggregation simulation using pairwise additive masking.

Protocol overview:
    1. Each pair of clients (i, j) agrees on a shared random seed.
    2. Client i adds PRG(seed_ij) to its update; client j subtracts the
       same PRG(seed_ij).
    3. When the server sums all masked updates, the pairwise masks cancel
       out perfectly — the server obtains the true aggregate without ever
       seeing any individual client's update.

Note: Secure aggregation protects individual updates in transit but does
NOT by itself provide a formal differential privacy guarantee.  It is a
complementary mechanism that can be combined with DP.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SecureAggregation:
    """
    Simulate the secure-aggregation protocol with pairwise additive masks.
    """

    def __init__(self, client_ids: List[str], random_state: int = 42):
        self.client_ids = sorted(client_ids)
        self.n_clients = len(client_ids)
        self.random_state = random_state
        self._pairwise_seeds: Dict[Tuple[str, str], int] = {}
        self._generate_pairwise_seeds()

    # ------------------------------------------------------------------
    # Seed / mask generation
    # ------------------------------------------------------------------

    def _generate_pairwise_seeds(self):
        """Generate a shared random seed for every ordered pair (i, j) with i < j."""
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_clients):
            for j in range(i + 1, self.n_clients):
                pair = (self.client_ids[i], self.client_ids[j])
                self._pairwise_seeds[pair] = int(rng.randint(0, 2**31))

    def compute_mask(self, client_id: str, shape: tuple) -> np.ndarray:
        """
        Compute the aggregate mask that *client_id* must add to its update.

        For every pair (i, j) with i < j:
            • client i  →  +PRG(seed_ij)
            • client j  →  −PRG(seed_ij)
        """
        mask = np.zeros(shape, dtype=np.float64)
        for i in range(self.n_clients):
            for j in range(i + 1, self.n_clients):
                pair = (self.client_ids[i], self.client_ids[j])
                seed = self._pairwise_seeds[pair]
                rng = np.random.RandomState(seed)
                pair_mask = rng.normal(0, 1.0, shape)

                if client_id == self.client_ids[i]:
                    mask += pair_mask
                elif client_id == self.client_ids[j]:
                    mask -= pair_mask
        return mask

    # ------------------------------------------------------------------
    # Masking and aggregation
    # ------------------------------------------------------------------

    def mask_update(
        self, client_id: str, update: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Add the client's mask to each parameter array."""
        return [param + self.compute_mask(client_id, param.shape) for param in update]

    def aggregate(
        self,
        masked_updates: Dict[str, List[np.ndarray]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[np.ndarray]:
        """
        Sum masked updates from all clients.

        Because pairwise masks cancel, the result is the (weighted) average
        of the *original* (unmasked) updates.

        Args:
            masked_updates: client_id → list of masked parameter arrays.
            weights: optional per-client weights (should sum to 1).

        Returns:
            Aggregated parameter list.
        """
        cids = list(masked_updates.keys())
        n_params = len(masked_updates[cids[0]])

        if weights is None:
            weights = {cid: 1.0 / len(cids) for cid in cids}

        aggregated = [np.zeros_like(masked_updates[cids[0]][p]) for p in range(n_params)]
        for cid in cids:
            for p in range(n_params):
                aggregated[p] += weights[cid] * masked_updates[cid][p]
        return aggregated

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_cancellation(self, shape: tuple = (10,)) -> bool:
        """
        Sanity-check: summing every client's mask must yield zero.
        Returns True if masks cancel within floating-point tolerance.
        """
        total = np.zeros(shape, dtype=np.float64)
        for cid in self.client_ids:
            total += self.compute_mask(cid, shape)
        return bool(np.allclose(total, 0, atol=1e-10))


def run_secure_aggregation_demo(
    client_updates: Dict[str, List[np.ndarray]],
    weights: Optional[Dict[str, float]] = None,
    random_state: int = 42,
) -> Dict:
    """
    Demonstrate secure aggregation on a set of client parameter updates.

    Args:
        client_updates: client_id → list of numpy arrays (model parameters).
        weights: optional per-client weights.
        random_state: RNG seed for pairwise mask generation.

    Returns:
        Dict with plain aggregate, secure aggregate, and verification flag.
    """
    client_ids = sorted(client_updates.keys())
    sa = SecureAggregation(client_ids, random_state=random_state)

    # Verify mask cancellation
    sample_shape = client_updates[client_ids[0]][0].shape
    masks_cancel = sa.verify_cancellation(sample_shape)

    # Plain (non-secure) aggregation for comparison
    n_params = len(client_updates[client_ids[0]])
    if weights is None:
        weights = {cid: 1.0 / len(client_ids) for cid in client_ids}

    plain_agg = [np.zeros_like(client_updates[client_ids[0]][p]) for p in range(n_params)]
    for cid in client_ids:
        for p in range(n_params):
            plain_agg[p] += weights[cid] * client_updates[cid][p]

    # Secure aggregation
    masked = {cid: sa.mask_update(cid, params) for cid, params in client_updates.items()}
    secure_agg = sa.aggregate(masked, weights)

    # Check equivalence
    matches = all(np.allclose(plain_agg[p], secure_agg[p], atol=1e-8) for p in range(n_params))

    return {
        'plain_aggregate': plain_agg,
        'secure_aggregate': secure_agg,
        'masks_cancel': masks_cancel,
        'aggregates_match': matches,
        'max_diff': max(
            np.max(np.abs(plain_agg[p] - secure_agg[p])) for p in range(n_params)
        ),
    }
