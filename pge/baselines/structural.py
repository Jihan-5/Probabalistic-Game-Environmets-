"""Direct structural statistics over the reasoning DAG (zero-shot).

The graphon models edge probabilities between node blocks; these detectors
instead read the hypothesized failure signature directly — contradicted
high-confidence commitments that are CONNECTED (error cascades) — without a
fitted model. If none of these beat per-node marginal baselines, the
environment carries no usable structural signal beyond marginals, and the
paper's framing must say so.
"""

from __future__ import annotations

import numpy as np

from pge.baselines.base import TrajectoryScorer


class NegNegEdgeFraction(TrajectoryScorer):
    """Fraction of DAG edges whose BOTH endpoints have contradicting evidence."""

    name = "negneg_edge_fraction"

    def score(self, trajectory, dag=None) -> float:
        if dag is None or not dag.edges:
            return 0.0
        s = dag.scores()
        negneg = sum(1 for p, c in dag.edges if s[p] < 0 and s[c] < 0)
        return negneg / len(dag.edges)


class CascadeLift(TrajectoryScorer):
    """Excess connectivity of contradicted nodes over the marginal rate.

    lift = P(both endpoints contradicted | edge) - P(contradicted)^2.
    Positive lift = contradicted nodes cluster along edges (cascade);
    a per-node detector cannot see this term.
    """

    name = "cascade_lift"

    def score(self, trajectory, dag=None) -> float:
        if dag is None or not dag.edges or len(dag) < 2:
            return 0.0
        s = dag.scores()
        neg_rate = float((s < 0).mean())
        negneg = sum(1 for p, c in dag.edges if s[p] < 0 and s[c] < 0)
        return negneg / len(dag.edges) - neg_rate**2


class ConfidentCascadeScore(TrajectoryScorer):
    """Weighted cascade mass: sum over edges of p_parent * p_child for
    contradicted-contradicted pairs, normalized by edge count. Combines the
    confidence and connectivity signatures of a hallucination cascade."""

    name = "confident_cascade"

    def score(self, trajectory, dag=None) -> float:
        if dag is None or not dag.edges:
            return 0.0
        s = dag.scores()
        p = dag.probabilities()
        mass = sum(
            p[i] * p[j] for i, j in dag.edges if s[i] < 0 and s[j] < 0
        )
        return float(mass / len(dag.edges))
