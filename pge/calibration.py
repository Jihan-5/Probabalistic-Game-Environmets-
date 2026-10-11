"""Temperature scaling for agent probability calibration (paper §3.1).

The paper uses temperature scaling on a 10% validation split to minimize
expected calibration error (ECE). Given raw probabilities p_i and binary
correctness labels y_i, find T > 0 minimizing ECE of the scaled
probabilities p_i^(1/T) / sum_k p_i^(1/T).

For our setup, where the agent emits a single probability per guess (not a
softmax over K classes), temperature scaling reduces to a logit-space
linear transform:

    p_scaled = sigmoid(logit(p) / T)

We find T by minimizing binary ECE on a held-out set of (probability,
correctness) pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Standard ECE: weighted average gap between bin confidence and bin accuracy."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask |= probs == 1.0
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_conf - bin_acc)
    return float(ece)


def fit_temperature(probs, labels, bracket=(0.05, 20.0)) -> float:
    """Find T > 0 that minimizes ECE on (probs, labels) via 1-D minimization."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) < 5:
        return 1.0

    def loss(T):
        if T <= 0:
            return 1.0
        scaled = _sigmoid(_logit(probs) / T)
        return expected_calibration_error(scaled, labels)

    res = minimize_scalar(loss, bracket=bracket, method="brent", options={"xtol": 1e-3})
    return float(res.x) if res.success else 1.0


def apply_temperature(probs, T: float) -> np.ndarray:
    """Apply learned temperature to a new set of raw probabilities."""
    probs = np.asarray(probs, dtype=float)
    return _sigmoid(_logit(probs) / T)
