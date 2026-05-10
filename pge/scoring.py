"""Bayesian behavioral scoring — paper Algorithm 2 / Eqs. 1-2.

Given an agent guess with calibrated probability p_t and an observed binary
behavior signal b_t ∈ {-1, +1}, produce a soft score s_t ∈ [-1, 1] via the
logit-then-tanh transform. The logit amplifies high-confidence guesses;
tanh keeps the output bounded for downstream graphon weighting.
"""

from __future__ import annotations

import numpy as np

_EPS = 0.01  # paper Algorithm 2 line 5: clip(p_t, 0.01, 0.99)


def bayesian_score(guess_probability: float, behavior_signal: int) -> float:
    """One-step score (Algorithm 2, Eq. 1).

    Args:
        guess_probability: calibrated probability p_t in (0, 1).
        behavior_signal: +1 if user behavior is consistent with the guess, -1 otherwise.

    Returns:
        s_t = tanh(b_t * log(p_t / (1 - p_t)))  in [-1, 1].
    """
    if behavior_signal not in (-1, 1):
        raise ValueError("behavior_signal must be -1 or +1")
    p = float(np.clip(guess_probability, _EPS, 1.0 - _EPS))
    logit = np.log(p / (1.0 - p))
    return float(np.tanh(behavior_signal * logit))


def cumulative_evidence(scores: np.ndarray | list[float]) -> tuple[float, float]:
    """Cumulative evidence and accumulated uncertainty (Eq. 2).

    Returns:
        (E_T, Var(E_T)) where E_T = sum(s_t) and Var(E_T) = sum(1 - |s_t|).
        The variance term grows with low-confidence (small-magnitude) scores.
    """
    s = np.asarray(scores, dtype=float)
    E_T = float(s.sum())
    var = float(np.sum(1.0 - np.abs(s)))
    return E_T, var
