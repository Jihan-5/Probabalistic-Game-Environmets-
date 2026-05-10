"""Sparse graphon estimation and likelihood — paper Section 3.3, Algorithm 3.

We treat each chain-of-thought DAG as a draw from an unknown graphon
W : [0,1]^2 -> [0,1]. We discretize into B blocks and estimate each block's
edge probability via the Beta(alpha, beta) posterior mean over training DAGs.

Detection: low score-weighted log-likelihood under the fitted graphon flags
a hallucinating reasoning trace.

G-Mixup (Eq. 6): convex combination of two graphons with lambda ~ Beta(0.4, 0.4)
for cross-domain interpolation; preserves homomorphism densities up to
O(m^2 * cut_distance) (paper Theorem 4.1).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from pge.dag import ChainOfThoughtDAG

_W_EPS = 1e-10  # Algorithm 3 line 6: clip(W, 1e-10, 1 - 1e-10)


def _block_index(node_idx: int, n: int, B: int) -> int:
    """Block assignment a_i = ceil((i / n) * B) - 1, in [0, B-1].

    The paper writes a_i = ceil(u_i * B) where u_i = i / |V_G|. We use the
    1-indexed-then-shift convention so that a_i lands in [0, B-1]. Node 0 is
    placed in block 0.
    """
    if n <= 1:
        return 0
    # u_i in (0, 1], block index in [0, B-1]
    u = (node_idx + 1) / n
    a = int(np.ceil(u * B)) - 1
    return max(0, min(B - 1, a))


class GraphonEstimator:
    """Block-model estimator for a sparse graphon (paper Eq. 3-4).

    Parameters
    ----------
    B : int
        Number of blocks (paper default: 10).
    alpha, beta : float
        Beta-prior hyperparameters for sparse regularization
        (paper default: alpha=0.1, beta=10).
    """

    def __init__(self, B: int = 10, alpha: float = 0.1, beta: float = 10.0):
        if B < 1:
            raise ValueError("B must be >= 1")
        self.B = B
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.W: Optional[np.ndarray] = None  # shape (B, B)

    # ------------------------------------------------------------------ fit

    def fit(self, dags: Iterable[ChainOfThoughtDAG]) -> "GraphonEstimator":
        """Estimate W from a collection of training DAGs.

        Eq. 3: N_ab = number of (i, j) edges with i in block a, j in block b.
        Eq. 4: D_ab = sum over training DAGs of |I_a| * |I_b|
                      = number of *possible* node pairs assignable to (a, b).
        Posterior mean with Beta(alpha, beta) prior:
                W[a, b] = (N_ab + alpha) / (D_ab + alpha + beta)
        """
        B = self.B
        N = np.zeros((B, B), dtype=float)
        D = np.zeros((B, B), dtype=float)

        for G in dags:
            n = len(G)
            if n < 2:
                continue
            blocks = np.array([_block_index(i, n, B) for i in range(n)])
            # Possible pairs per block-pair: count node-counts per block.
            counts = np.bincount(blocks, minlength=B)
            D += np.outer(counts, counts)
            for p, c in G.edges:
                a = blocks[p]
                b = blocks[c]
                N[a, b] += 1

        self.W = (N + self.alpha) / (D + self.alpha + self.beta)
        return self

    # --------------------------------------------------------- log-likelihood

    def log_likelihood(self, G: ChainOfThoughtDAG) -> float:
        """Score-weighted log-likelihood under W (paper Eq. 5, Algorithm 3).

        ell(G | W) = sum_{i<j} w_ij * [A_ij log W[a_i, a_j]
                                       + (1 - A_ij) log(1 - W[a_i, a_j])]

        with edge weights w_ij = sqrt(|s_i| * |s_j|) (paper Algorithm 3 line 7).
        """
        if self.W is None:
            raise RuntimeError("graphon not fit; call fit() first")
        n = len(G)
        if n < 2:
            return 0.0

        W = np.clip(self.W, _W_EPS, 1.0 - _W_EPS)
        A = G.adjacency()
        scores = np.abs(G.scores())
        blocks = np.array([_block_index(i, n, self.B) for i in range(n)])

        iu, ju = np.triu_indices(n, k=1)
        a = blocks[iu]
        b = blocks[ju]
        w = np.sqrt(scores[iu] * scores[ju])
        p = W[a, b]
        a_ij = A[iu, ju]

        ll = w * (a_ij * np.log(p) + (1 - a_ij) * np.log(1.0 - p))
        return float(ll.sum())


# ----------------------------------------------------------------- G-Mixup


def gmixup(
    W1: np.ndarray,
    W2: np.ndarray,
    lam: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """Graphon mixup (paper Eq. 6).

    W_mix = lam * W1 + (1 - lam) * W2, with lam ~ Beta(0.4, 0.4) by default.

    Returns:
        (W_mix, lam_used)
    """
    if W1.shape != W2.shape:
        raise ValueError(f"shape mismatch: {W1.shape} vs {W2.shape}")
    if lam is None:
        rng = rng if rng is not None else np.random.default_rng()
        lam = float(rng.beta(0.4, 0.4))
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return lam * W1 + (1.0 - lam) * W2, lam
