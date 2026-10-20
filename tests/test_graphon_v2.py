"""Tests for the v2 GraphonEstimator options: block methods, weighting, normalization."""

import numpy as np
import pytest

from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import BLOCK_METHODS, GraphonEstimator


def make_dag(n, probs=None, scores=None, edges=None):
    G = ChainOfThoughtDAG()
    for i in range(n):
        G.add_node(
            GuessNode(
                attribute=f"a{i}",
                value="v",
                probability=0.5 if probs is None else probs[i],
                score=0.5 if scores is None else scores[i],
                turn=i,
            )
        )
    if edges is None:
        edges = [(i - 1, i) for i in range(1, n)]
    for p, c in edges:
        G.add_edge(p, c)
    return G


def test_position_blocks_identical_for_fixed_length_chains():
    # The documented e0 failure mode: same n => same blocks regardless of content.
    est = GraphonEstimator(B=4, block_method="position")
    g1 = make_dag(6, scores=[0.9] * 6)
    g2 = make_dag(6, scores=[-0.9] * 6)
    assert np.array_equal(est.assign_blocks(g1), est.assign_blocks(g2))


def test_feature_blocks_separate_contradicted_nodes():
    est = GraphonEstimator(B=4, block_method="feature")
    g_ok = make_dag(6, probs=[0.9] * 6, scores=[0.9] * 6)
    g_bad = make_dag(6, probs=[0.9] * 6, scores=[-0.9] * 6)
    b_ok = est.assign_blocks(g_ok)
    b_bad = est.assign_blocks(g_bad)
    assert not np.array_equal(b_ok, b_bad)
    # contradicted (negative score) nodes land in the lower half of blocks
    assert b_bad.max() < est.B // 2 <= b_ok.min()


@pytest.mark.parametrize("method", BLOCK_METHODS)
def test_fit_and_likelihood_run_for_all_methods(method):
    rng = np.random.default_rng(0)
    dags = [
        make_dag(
            int(rng.integers(4, 12)),
            probs=list(rng.uniform(0.3, 0.95, 12)),
            scores=list(rng.uniform(-1, 1, 12)),
        )
        for _ in range(20)
    ]
    est = GraphonEstimator(B=4, block_method=method).fit(dags)
    lls = [est.log_likelihood(G) for G in dags]
    assert all(np.isfinite(ll) for ll in lls)


def test_signed_weighting_penalizes_conflicting_pairs():
    dags = [make_dag(6, scores=[0.8] * 6) for _ in range(10)]
    est_abs = GraphonEstimator(B=2, block_method="position", weighting="abs").fit(dags)
    est_sgn = GraphonEstimator(B=2, block_method="position", weighting="signed").fit(dags)
    mixed = make_dag(6, scores=[0.8, -0.8] * 3)
    clean = make_dag(6, scores=[0.8] * 6)
    # abs weighting can't tell them apart; signed can.
    assert est_abs.log_likelihood(mixed) == pytest.approx(est_abs.log_likelihood(clean))
    assert est_sgn.log_likelihood(mixed) != pytest.approx(est_sgn.log_likelihood(clean))


def test_normalization_makes_sizes_comparable():
    rng = np.random.default_rng(1)
    dags = [make_dag(n) for n in (4, 8, 16, 32)]
    est = GraphonEstimator(B=2, block_method="degree", normalize=True).fit(dags)
    small = est.log_likelihood(make_dag(4))
    large = est.log_likelihood(make_dag(32))
    # Per-pair scale: same chain topology should give same order of magnitude.
    assert abs(large) < abs(small) * 10


def test_breakdown_keys_and_consistency():
    dags = [make_dag(6) for _ in range(5)]
    est = GraphonEstimator(B=3).fit(dags)
    bd = est.log_likelihood_breakdown(dags[0])
    assert set(bd) == {"total", "structure", "mean_weight"}
    assert bd["total"] == pytest.approx(est.log_likelihood(dags[0]))


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        GraphonEstimator(block_method="nope")
    with pytest.raises(ValueError):
        GraphonEstimator(weighting="nope")
