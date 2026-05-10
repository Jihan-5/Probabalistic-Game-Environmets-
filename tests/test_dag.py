"""Tests for pge.dag."""

import numpy as np
import pytest

from pge.dag import ChainOfThoughtDAG, GuessNode


def _node(attribute="a", value=0, p=0.5, score=0.0, turn=0):
    return GuessNode(attribute=attribute, value=value, probability=p, score=score, turn=turn)


def test_add_node_returns_index():
    G = ChainOfThoughtDAG()
    i = G.add_node(_node())
    j = G.add_node(_node())
    assert i == 0 and j == 1
    assert len(G) == 2


def test_self_loops_rejected():
    G = ChainOfThoughtDAG()
    G.add_node(_node())
    with pytest.raises(ValueError):
        G.add_edge(0, 0)


def test_back_edges_rejected():
    G = ChainOfThoughtDAG()
    G.add_node(_node())
    G.add_node(_node())
    with pytest.raises(ValueError):
        G.add_edge(1, 0)


def test_out_of_range_edge_rejected():
    G = ChainOfThoughtDAG()
    G.add_node(_node())
    with pytest.raises(IndexError):
        G.add_edge(0, 5)


def test_adjacency_matrix_upper_triangular():
    G = ChainOfThoughtDAG()
    for _ in range(4):
        G.add_node(_node())
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(0, 3)
    A = G.adjacency()
    assert A.shape == (4, 4)
    # Upper-triangular (no entries on or below diagonal)
    assert np.all(np.tril(A) == 0)
    assert A[0, 1] == 1 and A[1, 2] == 1 and A[0, 3] == 1
    assert A[0, 2] == 0  # not added


def test_scores_and_probabilities_arrays():
    G = ChainOfThoughtDAG()
    G.add_node(_node(p=0.2, score=0.4))
    G.add_node(_node(p=0.7, score=-0.1))
    np.testing.assert_allclose(G.probabilities(), [0.2, 0.7])
    np.testing.assert_allclose(G.scores(), [0.4, -0.1])
