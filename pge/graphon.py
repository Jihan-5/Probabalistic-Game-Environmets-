"""Sparse graphon estimation and likelihood over chain-of-thought DAGs.

We treat each DAG as a draw from an unknown graphon W : [0,1]^2 -> [0,1],
discretized into B blocks; each block's edge probability is the Beta(alpha,
beta) posterior mean over training DAGs. Detection: low (score-weighted)
log-likelihood under the fitted graphon flags a hallucinating trace.

Rebuild notes (e0_diagnosis, results/RESULTS.md): the original estimator was
structurally blind — `block_method="position"` plus fixed-length sequential
chains gave every trajectory identical adjacency and blocks, and likelihood
weights used |score|, discarding the correct/incorrect sign. Confident-correct
and confident-hallucinated nodes contributed identically, so detection was at
chance. The fixes are selectable so the old behavior remains an honest
ablation arm:

    block_method : how nodes map to blocks
        "position"  — a_i = ceil((i+1)/n * B) - 1 (ORIGINAL; structure-blind
                      on fixed-length chains)
        "degree"    — rank nodes by symmetrized degree, blocks by rank
                      quantile (Chan & Airoldi sorted-degree smoothing)
        "spectral"  — rank by leading eigenvector of the symmetrized
                      adjacency, blocks by rank quantile
        "feature"   — block = (score sign) x (confidence bucket); lets W
                      encode how confident-contradicted commitments attach to
                      the rest of the reasoning graph. Needs node scores and
                      probabilities; the sign carries the behavioral outcome.
    weighting : "abs" (ORIGINAL, w_ij = sqrt(|s_i s_j|)) or "signed"
                (w_ij = s_i * s_j — pairs with conflicting evidence signs
                push the likelihood down instead of up)
    normalize : divide the log-likelihood by the number of node pairs so
                variable-size DAGs are comparable (required once T varies)

G-Mixup: convex combination of two graphons with lambda ~ Beta(0.4, 0.4)
for cross-domain interpolation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from pge.dag import ChainOfThoughtDAG

_W_EPS = 1e-10

BLOCK_METHODS = ("position", "degree", "spectral", "feature")
WEIGHTINGS = ("abs", "signed")


def _block_index(node_idx: int, n: int, B: int) -> int:
    """Positional block assignment a_i = ceil(((i+1)/n) * B) - 1, in [0, B-1]."""
    if n <= 1:
        return 0
    u = (node_idx + 1) / n
    a = int(np.ceil(u * B)) - 1
    return max(0, min(B - 1, a))


def _rank_to_blocks(order_values: np.ndarray, B: int) -> np.ndarray:
    """Assign blocks by rank quantile of `order_values` (stable ties)."""
    n = order_values.shape[0]
    ranks = np.empty(n, dtype=int)
    ranks[np.argsort(order_values, kind="stable")] = np.arange(n)
    return np.minimum((ranks * B) // max(n, 1), B - 1).astype(int)


class GraphonEstimator:
    """Block-model estimator for a sparse graphon.

    Parameters
    ----------
    B : int
        Number of blocks.
    alpha, beta : float
        Beta-prior hyperparameters for sparse regularization.
    block_method : str
        One of BLOCK_METHODS (see module docstring). Default "position"
        preserves the original behavior for ablations and old tests.
    weighting : str
        "abs" (original) or "signed".
    normalize : bool
        Length-normalize log-likelihoods by the number of node pairs.
    """

    def __init__(
        self,
        B: int = 10,
        alpha: float = 0.1,
        beta: float = 10.0,
        block_method: str = "position",
        weighting: str = "abs",
        normalize: bool = False,
    ):
        if B < 1:
            raise ValueError("B must be >= 1")
        if block_method not in BLOCK_METHODS:
            raise ValueError(f"block_method must be one of {BLOCK_METHODS}")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        self.B = B
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.block_method = block_method
        self.weighting = weighting
        self.normalize = bool(normalize)
        self.W: Optional[np.ndarray] = None  # shape (B, B)

    # -------------------------------------------------------- block assignment

    def assign_blocks(self, G: ChainOfThoughtDAG) -> np.ndarray:
        """Per-node block indices for DAG `G` under this estimator's method."""
        n = len(G)
        B = self.B
        if n == 0:
            return np.zeros(0, dtype=int)
        if self.block_method == "position":
            return np.array([_block_index(i, n, B) for i in range(n)])

        if self.block_method == "feature":
            # sign half x confidence bucket. B=1 collapses to a single block.
            n_conf = max(B // 2, 1)
            probs = np.clip(G.probabilities(), 0.0, 1.0)
            signs = (G.scores() >= 0).astype(int)  # 0 = contradicted, 1 = ok
            buckets = np.minimum((probs * n_conf).astype(int), n_conf - 1)
            blocks = signs * n_conf + buckets
            return np.minimum(blocks, B - 1).astype(int)

        A = G.adjacency()
        A_sym = ((A + A.T) > 0).astype(float)
        if self.block_method == "degree":
            return _rank_to_blocks(A_sym.sum(axis=1), B)

        # spectral: leading eigenvector of the symmetrized adjacency
        if n == 1:
            return np.zeros(1, dtype=int)
        _, vecs = np.linalg.eigh(A_sym)
        v = vecs[:, -1]
        if v.sum() < 0:  # sign convention for stability across runs
            v = -v
        return _rank_to_blocks(v, B)

    # ------------------------------------------------------------------ fit

    def fit(self, dags: Iterable[ChainOfThoughtDAG]) -> "GraphonEstimator":
        """Estimate W: posterior-mean block edge probabilities over `dags`."""
        B = self.B
        N = np.zeros((B, B), dtype=float)
        D = np.zeros((B, B), dtype=float)

        for G in dags:
            n = len(G)
            if n < 2:
                continue
            blocks = self.assign_blocks(G)
            counts = np.bincount(blocks, minlength=B)
            D += np.outer(counts, counts)
            for p, c in G.edges:
                N[blocks[p], blocks[c]] += 1

        self.W = (N + self.alpha) / (D + self.alpha + self.beta)
        return self

    # --------------------------------------------------------- log-likelihood

    def _pair_terms(self, G: ChainOfThoughtDAG):
        n = len(G)
        W = np.clip(self.W, _W_EPS, 1.0 - _W_EPS)
        A = G.adjacency()
        s = G.scores()
        blocks = self.assign_blocks(G)

        iu, ju = np.triu_indices(n, k=1)
        p = W[blocks[iu], blocks[ju]]
        a_ij = A[iu, ju]
        bern = a_ij * np.log(p) + (1 - a_ij) * np.log(1.0 - p)

        if self.weighting == "abs":
            w = np.sqrt(np.abs(s[iu]) * np.abs(s[ju]))
        else:  # signed
            w = s[iu] * s[ju]
        return bern, w

    def log_likelihood(self, G: ChainOfThoughtDAG) -> float:
        """Score-weighted log-likelihood of `G` under the fitted graphon."""
        if self.W is None:
            raise RuntimeError("graphon not fit; call fit() first")
        if len(G) < 2:
            return 0.0
        bern, w = self._pair_terms(G)
        total = float((w * bern).sum())
        if self.normalize:
            total /= bern.shape[0]
        return total

    def log_likelihood_breakdown(self, G: ChainOfThoughtDAG) -> dict:
        """Diagnostics: split the likelihood into structure and weight parts.

        Returns dict with `total` (as log_likelihood), `structure`
        (unweighted mean Bernoulli log-lik — pure adjacency/block signal) and
        `mean_weight` (average pair weight — pure score signal). If `total`
        tracks `mean_weight` across trajectories while `structure` is
        constant, the estimator is structurally blind (the e0 failure mode).
        """
        if self.W is None:
            raise RuntimeError("graphon not fit; call fit() first")
        if len(G) < 2:
            return {"total": 0.0, "structure": 0.0, "mean_weight": 0.0}
        bern, w = self._pair_terms(G)
        total = float((w * bern).sum())
        if self.normalize:
            total /= bern.shape[0]
        return {
            "total": total,
            "structure": float(bern.mean()),
            "mean_weight": float(w.mean()),
        }


# ----------------------------------------------------------------- G-Mixup


def gmixup(
    W1: np.ndarray,
    W2: np.ndarray,
    lam: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """W_mix = lam * W1 + (1 - lam) * W2, lam ~ Beta(0.4, 0.4) by default."""
    if W1.shape != W2.shape:
        raise ValueError(f"shape mismatch: {W1.shape} vs {W2.shape}")
    if lam is None:
        rng = rng if rng is not None else np.random.default_rng()
        lam = float(rng.beta(0.4, 0.4))
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return lam * W1 + (1.0 - lam) * W2, lam
