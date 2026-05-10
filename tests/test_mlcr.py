"""Tests for pge.mlcr (paper Algorithm 4 / Section 3.4)."""

import numpy as np
import pytest

from pge.constraints import SimpleConstraints
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator
from pge.mlcr import mlcr_resolve


def _toy_dag(operations):
    G = ChainOfThoughtDAG()
    for t, op in enumerate(operations):
        G.add_node(GuessNode(
            attribute=f"a{t}", value=op, probability=0.7, score=0.6,
            turn=t, operation=op,
        ))
    for t in range(1, len(operations)):
        G.add_edge(t - 1, t)
    return G


def _toy_graphon(B=3):
    rng = np.random.default_rng(0)
    train = []
    for _ in range(20):
        train.append(_toy_dag([0, 1, 0, 1, 0]))
    return GraphonEstimator(B=B).fit(train)


def test_mlcr_returns_one_op_per_node():
    G = _toy_dag(["A", "B", "A", "B", "A"])
    est = _toy_graphon()
    cands = [["A", "B"]] * 5
    constraints = SimpleConstraints(candidates=cands)
    out = mlcr_resolve(G, est, constraints)
    assert len(out) == 5
    for op in out:
        assert op in {"A", "B"}


def test_mlcr_respects_feasibility():
    """With node 0 restricted to {'X'}, MLCR must return 'X' there."""
    G = _toy_dag(["A", "B", "A", "B", "A"])
    est = _toy_graphon()
    cands = [["X"], ["A", "B"], ["A", "B"], ["A", "B"], ["A", "B"]]
    constraints = SimpleConstraints(candidates=cands)
    out = mlcr_resolve(G, est, constraints)
    assert out[0] == "X"


def test_mlcr_terminates_within_max_iter():
    """Convergence guarantee: should finish in at most max_iter iterations.

    We can't observe iteration count directly, but we can run with max_iter=1
    and confirm the call returns (i.e. doesn't loop forever).
    """
    G = _toy_dag(["A", "B", "A", "B", "A"])
    est = _toy_graphon()
    constraints = SimpleConstraints(candidates=[["A", "B"]] * 5)
    out = mlcr_resolve(G, est, constraints, max_iter=1)
    assert len(out) == 5


def test_mlcr_uses_op_score_to_prefer_specific_op():
    G = _toy_dag(["A", "A", "A"])
    est = _toy_graphon()
    constraints = SimpleConstraints(candidates=[["A", "B"]] * 3)
    # Strongly prefer B at every node.
    op_score = {i: {"A": 0.0, "B": 100.0} for i in range(3)}
    out = mlcr_resolve(G, est, constraints, op_score=op_score)
    assert out == ["B", "B", "B"]


def test_mlcr_empty_dag():
    G = ChainOfThoughtDAG()
    est = _toy_graphon()
    constraints = SimpleConstraints(candidates=[])
    assert mlcr_resolve(G, est, constraints) == []
