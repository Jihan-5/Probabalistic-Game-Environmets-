"""DAG-feature logistic-regression detector.

Serves two roles:
  1. A strong supervised baseline over hand-crafted graph features.
  2. The FAIR ablation arm "full PGE minus graphon" —
     it consumes the same DAG (same scoring, same edges) but summarizes it
     with per-node marginal features instead of graphon likelihood. What it
     cannot see, by construction, is edge-level error correlation (cascade
     structure) beyond simple counts — exactly the graphon's claimed value.

Pure numpy (no sklearn dependency): L2-regularized logistic regression via
full-batch gradient descent — the feature dimension is ~14, this is plenty.
"""

from __future__ import annotations

import numpy as np

from pge.baselines.base import TrajectoryScorer
from pge.dag import ChainOfThoughtDAG


def dag_feature_vector(G: ChainOfThoughtDAG) -> np.ndarray:
    n = max(len(G), 1)
    probs = G.probabilities() if len(G) else np.zeros(1)
    scores = G.scores() if len(G) else np.zeros(1)
    n_edges = len(G.edges)
    neg = scores < 0
    high_conf = probs >= 0.6

    # longest run of consecutive (by node index) negative scores
    longest_neg_run = run = 0
    for is_neg in neg:
        run = run + 1 if is_neg else 0
        longest_neg_run = max(longest_neg_run, run)

    out_deg = np.zeros(n)
    in_deg = np.zeros(n)
    for p, c in G.edges:
        out_deg[p] += 1
        in_deg[c] += 1

    return np.array(
        [
            n,
            n_edges,
            n_edges / (n * (n - 1) / 2 + 1e-9),  # density
            probs.mean(),
            probs.max(initial=0.0),
            probs.std(),
            scores.mean(),
            scores.min(initial=0.0),
            float(neg.mean()),
            float((neg & high_conf).mean()),  # confident-contradicted fraction
            float(longest_neg_run) / n,
            out_deg.max(initial=0.0),
            in_deg.max(initial=0.0),
            float((np.abs(scores) < 0.2).mean()),  # unobserved/uncertain mass
        ],
        dtype=float,
    )


class DagFeatureClassifier(TrajectoryScorer):
    name = "dag_logreg"

    def __init__(self, l2: float = 1e-2, lr: float = 0.1, iters: int = 3000):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, trajectories, dags=None, labels=None):
        if dags is None or labels is None:
            raise ValueError("dag_logreg requires dags and labels to fit")
        X = np.stack([dag_feature_vector(G) for G in dags])
        y = np.asarray(labels, dtype=float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xz = (X - self.mu) / self.sd

        rng = np.random.default_rng(0)
        w = rng.normal(0, 0.01, Xz.shape[1])
        b = 0.0
        for _ in range(self.iters):
            z = Xz @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = p - y
            gw = Xz.T @ g / len(y) + self.l2 * w
            gb = float(g.mean())
            w -= self.lr * gw
            b -= self.lr * gb
        self.w, self.b = w, b
        return self

    def score(self, trajectory, dag=None) -> float:
        if self.w is None:
            raise RuntimeError("fit first")
        if dag is None:
            raise ValueError("dag_logreg requires the DAG")
        x = (dag_feature_vector(dag) - self.mu) / self.sd
        z = float(x @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
