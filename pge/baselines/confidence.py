"""Trivial confidence/evidence baselines (zero-shot).

These are the floor every structural method must beat. They consume only
detector-visible channels: reported confidences and the non-circular
behavior signals (e.g. healthcare v2's `noisy_feedback`).
"""

from __future__ import annotations

import numpy as np

from pge.baselines.base import TrajectoryScorer
from pge.scoring import bayesian_score, cumulative_evidence
from pge.trajectory import Trajectory


def _signals(traj: Trajectory) -> list[int]:
    out = []
    for t in traj.turns:
        # one named signal channel per benchmark; combine if several
        vals = [v for v in t.signals.values()]
        out.append(int(np.sign(sum(vals))) if vals else 0)
    return out


class MeanConfidence(TrajectoryScorer):
    """Hallucination score = mean reported confidence (overconfidence proxy)."""

    name = "mean_confidence"

    def score(self, trajectory, dag=None) -> float:
        return float(np.mean([t.probability for t in trajectory.turns]))


class MaxConfidence(TrajectoryScorer):
    name = "max_confidence"

    def score(self, trajectory, dag=None) -> float:
        return float(np.max([t.probability for t in trajectory.turns]))


class NegCumulativeEvidence(TrajectoryScorer):
    """-E_T over the observed (non-circular) signal channel."""

    name = "neg_cum_evidence"

    def score(self, trajectory, dag=None) -> float:
        scores = [
            bayesian_score(t.probability, s)
            for t, s in zip(trajectory.turns, _signals(trajectory))
        ]
        E_T, _ = cumulative_evidence(scores)
        return -E_T


class NegativeSignalFraction(TrajectoryScorer):
    """Fraction of observed signals that contradict the agent."""

    name = "neg_signal_fraction"

    def score(self, trajectory, dag=None) -> float:
        sig = [s for s in _signals(trajectory) if s != 0]
        if not sig:
            return 0.0
        return float(sum(1 for s in sig if s < 0) / len(sig))


class HighConfContradicted(TrajectoryScorer):
    """Fraction of high-confidence commitments with a contradicting signal —
    the per-node marginal version of what the graphon should capture
    structurally. If the graphon can't beat this, structure adds nothing."""

    name = "highconf_contradicted"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def score(self, trajectory, dag=None) -> float:
        sig = _signals(trajectory)
        high = [
            (t, s)
            for t, s in zip(trajectory.turns, sig)
            if t.probability >= self.threshold and s != 0
        ]
        if not high:
            return 0.0
        return float(sum(1 for _, s in high if s < 0) / len(high))
