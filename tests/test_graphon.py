"""Tests for pge.graphon (paper Section 3.3, Algorithm 3, Eq. 6)."""

import numpy as np
import pytest

from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator, gmixup


def _make_chain_dag(n: int, edge_prob: float, rng: np.random.Generator) -> ChainOfThoughtDAG:
    G = ChainOfThoughtDAG()
    for t in range(n):
        G.add_node(GuessNode(
            attribute=f"a{t}", value=t, probability=0.7, score=0.7, turn=t
        ))
    # Sequential dependencies + sparse random shortcuts
    for t in range(1, n):
        G.add_edge(t - 1, t)
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
    return G


def test_fit_produces_W_in_unit_interval():
    rng = np.random.default_rng(0)
    dags = [_make_chain_dag(8, 0.2, rng) for _ in range(20)]
    est = GraphonEstimator(B=4).fit(dags)
    assert est.W is not None
    assert est.W.shape == (4, 4)
    assert (est.W >= 0).all() and (est.W <= 1).all()


def test_in_distribution_likelihood_higher_than_random():
    rng = np.random.default_rng(1)
    train = [_make_chain_dag(10, 0.25, rng) for _ in range(50)]
    est = GraphonEstimator(B=5).fit(train)

    # In-distribution test DAGs
    in_dags = [_make_chain_dag(10, 0.25, rng) for _ in range(20)]
    # Out-of-distribution: random pairs unrelated to position-based blocks
    out_dags = []
    for _ in range(20):
        G = ChainOfThoughtDAG()
        for t in range(10):
            G.add_node(GuessNode(
                attribute=f"a{t}", value=t, probability=0.7, score=0.7, turn=t
            ))
        # Densest near the diagonal in a way the train set never produces:
        # add many long-range edges, no sequential ones.
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.7:
                    G.add_edge(i, j)
        out_dags.append(G)

    in_ll = np.mean([est.log_likelihood(G) for G in in_dags])
    out_ll = np.mean([est.log_likelihood(G) for G in out_dags])
    assert in_ll > out_ll


def test_likelihood_requires_fit():
    G = ChainOfThoughtDAG()
    G.add_node(GuessNode(attribute="a", value=0, probability=0.5, score=0.5, turn=0))
    G.add_node(GuessNode(attribute="b", value=1, probability=0.5, score=0.5, turn=1))
    G.add_edge(0, 1)
    with pytest.raises(RuntimeError):
        GraphonEstimator(B=3).log_likelihood(G)


def test_gmixup_convex_combination_in_unit_interval():
    rng = np.random.default_rng(2)
    W1 = rng.uniform(0, 1, size=(5, 5))
    W2 = rng.uniform(0, 1, size=(5, 5))
    W_mix, lam = gmixup(W1, W2, lam=0.3)
    assert lam == pytest.approx(0.3)
    np.testing.assert_allclose(W_mix, 0.3 * W1 + 0.7 * W2)
    assert (W_mix >= 0).all() and (W_mix <= 1).all()


def test_gmixup_default_lambda_in_range():
    rng = np.random.default_rng(3)
    W1 = rng.uniform(0, 1, size=(3, 3))
    W2 = rng.uniform(0, 1, size=(3, 3))
    for _ in range(20):
        _, lam = gmixup(W1, W2, rng=rng)
        assert 0.0 <= lam <= 1.0


def test_gmixup_shape_mismatch():
    with pytest.raises(ValueError):
        gmixup(np.zeros((3, 3)), np.zeros((4, 4)))
