"""Graph-convolutional detector over the reasoning DAG (supervised).

Failure mode targeted: hallucination shows up as *correlated* evidence
against connected guesses — cascades of contradicted commitments along DAG
edges — which per-node marginals (dag_logreg) summarize but cannot
propagate. A 2-layer GCN lets each node see its neighborhood's
probability/score pattern before pooling to a trajectory-level logit.

Pure numpy, no torch: symmetric-normalized propagation
A_hat = D^-1/2 (A + A^T + I) D^-1/2, two ReLU graph-conv layers (hidden 16),
mean-pool, logistic head; full-batch Adam, seeded init. Variable-size
graphs are handled natively (per-graph forward/backward, no padding).
"""

from __future__ import annotations

import numpy as np

from pge.baselines.base import TrajectoryScorer
from pge.dag import ChainOfThoughtDAG

_UNCERTAIN_BAND = 0.2  # |score| below this = effectively unobserved node


def _node_features(G: ChainOfThoughtDAG) -> np.ndarray:
    """Per-node features: [probability, score, turn/max(T,1), |score|<band]."""
    n = len(G)
    if n == 0:
        return np.zeros((0, 4), dtype=float)
    T = max(nd.turn for nd in G.nodes)
    return np.array(
        [
            [
                nd.probability,
                nd.score,
                nd.turn / max(T, 1),
                1.0 if abs(nd.score) < _UNCERTAIN_BAND else 0.0,
            ]
            for nd in G.nodes
        ],
        dtype=float,
    )


def _norm_adjacency(G: ChainOfThoughtDAG) -> np.ndarray:
    """A_hat = D^-1/2 (A + A^T + I) D^-1/2 (symmetrized, self-loops)."""
    n = len(G)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    A = G.adjacency().astype(float)
    A = A + A.T + np.eye(n)
    d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1))
    return A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


class GCNClassifier(TrajectoryScorer):
    name = "gcn"

    def __init__(
        self,
        hidden: int = 16,
        epochs: int = 300,
        lr: float = 0.01,
        l2: float = 1e-4,
        seed: int = 0,
    ):
        self.hidden, self.epochs, self.lr, self.l2, self.seed = (
            hidden,
            epochs,
            lr,
            l2,
            seed,
        )
        self.params: dict[str, np.ndarray] | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    # ------------------------------------------------------------- forward

    def _forward(self, A: np.ndarray, X: np.ndarray):
        p = self.params
        if X.shape[0] == 0:
            h = np.zeros(self.hidden)
            zero = np.zeros((0, self.hidden))
            return _sigmoid(float(h @ p["w3"]) + float(p["b3"][0])), (
                A, X, zero, zero, zero, zero, h,
            )
        Z1 = A @ X @ p["W1"] + p["b1"]
        H1 = np.maximum(Z1, 0.0)
        Z2 = A @ H1 @ p["W2"] + p["b2"]
        H2 = np.maximum(Z2, 0.0)
        h = H2.mean(axis=0)
        return _sigmoid(float(h @ p["w3"]) + float(p["b3"][0])), (
            A, X, Z1, H1, Z2, H2, h,
        )

    # ------------------------------------------------------------- fit

    def fit(self, trajectories, dags=None, labels=None):
        if dags is None or labels is None:
            raise ValueError("gcn requires dags and labels to fit")
        y = np.asarray(labels, dtype=float)
        feats = [_node_features(G) for G in dags]
        adjs = [_norm_adjacency(G) for G in dags]

        stacked = np.vstack([X for X in feats if X.shape[0] > 0])
        self.mu = stacked.mean(axis=0)
        self.sd = stacked.std(axis=0) + 1e-9
        Xs = [(X - self.mu) / self.sd if X.shape[0] else X for X in feats]

        rng = np.random.default_rng(self.seed)
        H = self.hidden
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / 4), (4, H)),
            "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(2.0 / H), (H, H)),
            "b2": np.zeros(H),
            "w3": rng.normal(0, np.sqrt(1.0 / H), H),
            "b3": np.zeros(1),
        }
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(vv) for k, vv in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = len(y)

        for epoch in range(1, self.epochs + 1):
            grads = {k: np.zeros_like(p) for k, p in self.params.items()}
            for A, X, yi in zip(adjs, Xs, y):
                pred, (_, _, Z1, H1, Z2, _, h) = self._forward(A, X)
                dz = pred - yi  # dL/dz for BCE + sigmoid
                grads["w3"] += dz * h
                grads["b3"] += np.array([dz])
                if X.shape[0] == 0:
                    continue
                dh = dz * self.params["w3"]
                dH2 = np.tile(dh / X.shape[0], (X.shape[0], 1))
                dZ2 = dH2 * (Z2 > 0)
                AH1 = A @ H1
                grads["W2"] += AH1.T @ dZ2
                grads["b2"] += dZ2.sum(axis=0)
                dH1 = A.T @ dZ2 @ self.params["W2"].T
                dZ1 = dH1 * (Z1 > 0)
                AX = A @ X
                grads["W1"] += AX.T @ dZ1
                grads["b1"] += dZ1.sum(axis=0)

            for k in self.params:
                g = grads[k] / n + self.l2 * self.params[k]
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * g * g
                m_hat = m[k] / (1 - beta1**epoch)
                v_hat = v[k] / (1 - beta2**epoch)
                self.params[k] = self.params[k] - self.lr * m_hat / (
                    np.sqrt(v_hat) + eps
                )
        return self

    # ------------------------------------------------------------- score

    def score(self, trajectory, dag=None) -> float:
        if self.params is None:
            raise RuntimeError("fit first")
        if dag is None:
            raise ValueError("gcn requires the DAG")
        X = _node_features(dag)
        if X.shape[0]:
            X = (X - self.mu) / self.sd
        pred, _ = self._forward(_norm_adjacency(dag), X)
        return pred
