"""Commitment scoring: Bayesian evidence scores and asymmetric proper-ish scoring.

Two scoring families live here:

1. `bayesian_score` / `cumulative_evidence` — the original evidence-accumulation
   scores (logit-then-tanh), kept as the DAG node score and as an ablation arm.

2. Asymmetric commitment scoring (the corrected scoring theory).
   A commitment (p_hat, y) with reported confidence p_hat and outcome
   y in {+1 consistent, -1 inconsistent, 0 unobserved} receives

       S(p_hat, y) =  alpha * p_hat          if y = +1
                     -beta  * p_hat ** gamma if y = -1
                      0                      if y =  0   (no signal)

   with alpha=1, beta=3, gamma=2 defaults. This scoring rule is NOT
   incentive-compatible in the exact-calibration sense: the
   expected-score-maximizing report for true belief q solves

       d/dp [ q*alpha*p - (1-q)*beta*p**gamma ] = 0
       =>  p*(q) = [ alpha*q / (beta*gamma*(1-q)) ] ** (1/(gamma-1))

   The corrected lemma we implement and test (NOTE: p* < q does NOT hold
   "for all q in (0,1)" — the exact statement, for gamma = 2, is):

       Conservative-report lemma (gamma = 2): p*(q) = alpha*q /
       (2*beta*(1-q)) is strictly increasing in q up to the saturation
       point q_sat = beta*gamma / (alpha + beta*gamma) (p* = 1 for
       q >= q_sat), and CONSERVATIVE (p* < q) exactly on
       q in (0, 1 - alpha/(beta*gamma)). With the defaults (1, 3, 2) that
       is q < 5/6; on (5/6, 6/7) the optimal report mildly EXCEEDS q.
       `implied_belief` is the exact inverse on the interior branch, so
       belief ranking is preserved and the true belief is recoverable from
       the report. Threshold-based intervention therefore operates on
       de-biased beliefs, not raw reports.

   `revisitation_threshold` derives the intervention trigger from the same
   primitives: the expected cumulative score of an agent whose per-commitment
   accuracy is q_min, reporting optimally. Falling below it means the agent
   is doing worse than a minimally-competent honest agent — revisit.
"""

from __future__ import annotations

import numpy as np

_EPS = 0.01  # clip(p_t, 0.01, 0.99)

# Default asymmetric-scoring parameters (paper Section 3.2).
ALPHA = 1.0
BETA = 3.0
GAMMA = 2.0


# --------------------------------------------------------------------------
# Original Bayesian evidence scoring (kept: DAG node scores + ablation arm)
# --------------------------------------------------------------------------


def bayesian_score(guess_probability: float, behavior_signal: int) -> float:
    """One-step evidence score s_t = tanh(b_t * logit(p_t)) in [-1, 1].

    behavior_signal: +1 consistent, -1 inconsistent, 0 unobserved (-> 0.0).
    """
    if behavior_signal not in (-1, 0, 1):
        raise ValueError("behavior_signal must be -1, 0, or +1")
    if behavior_signal == 0:
        return 0.0
    p = float(np.clip(guess_probability, _EPS, 1.0 - _EPS))
    logit = np.log(p / (1.0 - p))
    return float(np.tanh(behavior_signal * logit))


def cumulative_evidence(scores: np.ndarray | list[float]) -> tuple[float, float]:
    """(E_T, Var(E_T)) with E_T = sum(s_t), Var = sum(1 - |s_t|)."""
    s = np.asarray(scores, dtype=float)
    E_T = float(s.sum())
    var = float(np.sum(1.0 - np.abs(s)))
    return E_T, var


# --------------------------------------------------------------------------
# Asymmetric commitment scoring (corrected theory)
# --------------------------------------------------------------------------


def asymmetric_score(
    p_hat: float,
    outcome: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Asymmetric commitment score: rewards linear in confidence when right,
    penalizes super-linearly (p^gamma) when wrong. outcome in {-1, 0, +1}."""
    if outcome not in (-1, 0, 1):
        raise ValueError("outcome must be -1, 0, or +1")
    p = float(np.clip(p_hat, 0.0, 1.0))
    if outcome == 0:
        return 0.0
    if outcome == 1:
        return alpha * p
    return -beta * p**gamma


def optimal_report(
    q: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Expected-score-maximizing report p*(q) for true belief q (conservative-
    report lemma). Clipped to [0, 1]; strictly monotone in q on the interior."""
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for a unique interior optimum")
    q = float(np.clip(q, 0.0, 1.0))
    if q >= 1.0:
        return 1.0
    if q <= 0.0:
        return 0.0
    p_star = (alpha * q / (beta * gamma * (1.0 - q))) ** (1.0 / (gamma - 1.0))
    return float(min(1.0, p_star))


def implied_belief(
    p_hat: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Exact inverse of `optimal_report` on the interior: recover the belief q
    that would make p_hat the optimal report.

        q_hat = beta*gamma*p_hat^(gamma-1) / (alpha + beta*gamma*p_hat^(gamma-1))

    Apply this de-biasing BEFORE temperature calibration so downstream
    thresholds operate on beliefs, not conservative reports.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1")
    p = float(np.clip(p_hat, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    z = beta * gamma * p ** (gamma - 1.0)
    return float(z / (alpha + z))


def expected_score(
    q: float,
    p_hat: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """E[S] for a commitment with true correctness prob q and report p_hat."""
    q = float(np.clip(q, 0.0, 1.0))
    p = float(np.clip(p_hat, 0.0, 1.0))
    return q * alpha * p - (1.0 - q) * beta * p**gamma


def revisitation_threshold(
    q_min: float,
    t: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Theory-derived revisitation trigger after t scored commitments.

    tau_rev(t) = t * E[S | q_min, p*(q_min)]: the expected cumulative
    asymmetric score of an agent with per-commitment accuracy q_min playing
    the optimal conservative report. A running score below this bound means
    the agent is underperforming a minimally-competent honest agent, and the
    chain should be revisited. No tuned constants.
    """
    if t < 0:
        raise ValueError("t must be >= 0")
    p_star = optimal_report(q_min, alpha, beta, gamma)
    return float(t) * expected_score(q_min, p_star, alpha, beta, gamma)
