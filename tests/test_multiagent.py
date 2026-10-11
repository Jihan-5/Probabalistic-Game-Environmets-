"""Tests for pge.multiagent (commitments, conflicts, resolvers)."""

import numpy as np
import pytest

from pge.benchmarks.healthcare_v2 import sample_session_v2
from pge.graphon import GraphonEstimator
from pge.multiagent import (
    BestSingleAgent,
    Commitment,
    Conflict,
    build_merged_dag,
    collect_commitments,
    confidence_weighted_vote,
    find_conflicts,
    majority_vote,
    mlcr_resolution,
)


def _c(agent, attr, value, p):
    return Commitment(agent_id=agent, attribute=attr, value=value, probability=p)


# ------------------------------------------------------------- collection


def test_collect_commitments_one_per_agent_per_attr():
    rng = np.random.default_rng(0)
    session = sample_session_v2(rng, 0)
    profiles = [
        {"agent_id": "a0", "hallucinator": False, "noise": 0.05},
        {"agent_id": "a1", "hallucinator": True, "noise": 0.05},
    ]
    by_attr = collect_commitments(session, profiles, rng)
    assert set(by_attr) == set(session.schedule)
    for attr, comms in by_attr.items():
        assert [c.agent_id for c in comms] == ["a0", "a1"]
        for c in comms:
            assert c.attribute == attr
            assert 0.05 <= c.probability <= 0.95


# ------------------------------------------------------- conflict detection


def test_no_conflict_when_values_agree():
    by_attr = {"x": [_c("a0", "x", "yes", 0.9), _c("a1", "x", "yes", 0.85)]}
    assert find_conflicts(by_attr) == []


def test_no_conflict_when_disagreement_is_low_confidence():
    by_attr = {"x": [_c("a0", "x", "yes", 0.9), _c("a1", "x", "no", 0.4)]}
    assert find_conflicts(by_attr, p_min=0.7) == []


def test_conflict_on_confident_disagreement():
    by_attr = {
        "x": [_c("a0", "x", "yes", 0.9), _c("a1", "x", "no", 0.8)],
        "y": [_c("a0", "y", "low", 0.9), _c("a1", "y", "low", 0.9)],
    }
    conflicts = find_conflicts(by_attr, p_min=0.7)
    assert len(conflicts) == 1
    assert conflicts[0].attribute == "x"
    assert len(conflicts[0].commitments) == 2  # ALL commitments retained


# ----------------------------------------------------------------- voting


def test_majority_vote_plurality_and_tiebreak():
    cf = Conflict(
        "x",
        [
            _c("a0", "x", "yes", 0.71),
            _c("a1", "x", "yes", 0.72),
            _c("a2", "x", "no", 0.95),
        ],
    )
    assert majority_vote(cf) == "yes"
    # 1-1 tie -> higher summed probability wins.
    tie = Conflict("x", [_c("a0", "x", "yes", 0.9), _c("a1", "x", "no", 0.8)])
    assert majority_vote(tie) == "yes"


def test_confidence_weighted_vote_can_override_majority():
    # Two very weak supporters vs one strong: implied-belief mass favors "no".
    cf = Conflict(
        "x",
        [
            _c("a0", "x", "yes", 0.05),
            _c("a1", "x", "yes", 0.05),
            _c("a2", "x", "no", 0.95),
        ],
    )
    assert majority_vote(cf) == "yes"
    assert confidence_weighted_vote(cf) == "no"


# ------------------------------------------------------------- merged DAG


def _toy_commitments():
    schedule = ["p", "q", "r"]
    deps = {
        "p": (["yes", "no"], []),
        "q": (["yes", "no"], ["attr:p"]),
        "r": (["a", "b"], []),
    }
    by_attr = {
        "p": [_c("a0", "p", "yes", 0.9), _c("a1", "p", "yes", 0.8), _c("a2", "p", "yes", 0.85)],
        "q": [_c("a0", "q", "yes", 0.9), _c("a1", "q", "no", 0.8), _c("a2", "q", "yes", 0.85)],
        "r": [_c("a0", "r", "a", 0.6), _c("a1", "r", "a", 0.7), _c("a2", "r", "a", 0.65)],
    }
    return schedule, deps, by_attr


def test_merged_dag_structure():
    schedule, deps, by_attr = _toy_commitments()
    G = build_merged_dag(schedule, by_attr, dependencies=deps)
    n_agents, T = 3, 3
    assert len(G) == n_agents * T
    # All edges strictly forward.
    for p, c in G.edges:
        assert p < c
    # Per-agent dependency edge p -> q at each agent's offset.
    for k in range(n_agents):
        assert (k * T + 0, k * T + 1) in G.edges
    # Cross-agent same-attribute edges for every attribute and agent pair.
    for t in range(T):
        for k1 in range(n_agents):
            for k2 in range(k1 + 1, n_agents):
                assert (k1 * T + t, k2 * T + t) in G.edges
    # Operations carry the committed values.
    assert G.nodes[1].operation == "yes" and G.nodes[T + 1].operation == "no"


# ------------------------------------------------------------------- MLCR


def test_mlcr_resolution_end_to_end():
    schedule, deps, by_attr = _toy_commitments()
    merged = build_merged_dag(schedule, by_attr, dependencies=deps)
    graphon = GraphonEstimator(
        B=4, block_method="feature", weighting="abs", normalize=True
    ).fit([merged] * 10)
    conflicts = find_conflicts(by_attr, p_min=0.7)
    assert [cf.attribute for cf in conflicts] == ["q"]
    resolved = mlcr_resolution(merged, conflicts, graphon)
    assert set(resolved) == {"q"}
    assert resolved["q"] in {"yes", "no"}  # from the proposed set


# --------------------------------------------------------- best single agent


def test_best_single_agent_fit_predict_no_leakage():
    # Train: a1 is always right, a0 always wrong, a2 mixed.
    train = [
        Conflict("x", [_c("a0", "x", "no", 0.9), _c("a1", "x", "yes", 0.8), _c("a2", "x", "yes", 0.8)]),
        Conflict("y", [_c("a0", "y", "no", 0.9), _c("a1", "y", "yes", 0.8), _c("a2", "y", "no", 0.8)]),
    ]
    truths = {"x": "yes", "y": "yes"}
    model = BestSingleAgent().fit(train, lambda cf: truths[cf.attribute])
    assert model.best_agent_id == "a1"
    assert model.train_accuracy == 1.0
    # Test: predict returns a1's value even when a1 is in the minority —
    # test-time truth is never consulted.
    test_cf = Conflict(
        "z",
        [_c("a0", "z", "no", 0.9), _c("a1", "z", "yes", 0.8), _c("a2", "z", "no", 0.8)],
    )
    assert model.predict(test_cf) == "yes"
    # Fallback to majority when the chosen agent is absent.
    absent = Conflict("w", [_c("a0", "w", "no", 0.9), _c("a2", "w", "no", 0.8)])
    assert model.predict(absent) == "no"


def test_best_single_agent_requires_fit():
    cf = Conflict("x", [_c("a0", "x", "yes", 0.9)])
    with pytest.raises(RuntimeError):
        BestSingleAgent().predict(cf)


def test_reliability_mlcr_fit_and_resolve():
    """ReliabilityMLCR separates voters via the cascade bit (disagreement
    with consensus on parent attributes), downweighting the off-track
    coalition instead of inverting it."""
    from pge.multiagent.resolve import ReliabilityMLCR
    from pge.multiagent.commitments import Commitment
    from pge.multiagent.conflict import find_conflicts

    def commits(attr, triples):
        return [Commitment(a, attr, v, p) for a, v, p in triples]

    def session(a1_parent_off: bool, a1_child_val: str):
        # renal_screening_required depends on attr:needs_contrast.
        parent_val_a1 = "no" if a1_parent_off else "yes"
        return {
            "needs_contrast": commits(
                "needs_contrast",
                [("a0", "yes", 0.85), ("a1", parent_val_a1, 0.9), ("a2", "yes", 0.8)],
            ),
            "renal_screening_required": commits(
                "renal_screening_required",
                [("a0", "yes", 0.8), ("a1", a1_child_val, 0.92), ("a2", "yes", 0.75)],
            ),
        }

    # Train: a1 is off-track on the parent and wrong on the child.
    train_items = []
    for _ in range(40):
        by_attr = session(a1_parent_off=True, a1_child_val="no")
        for cf in find_conflicts(by_attr, p_min=0.7):
            truth = {"needs_contrast": "yes", "renal_screening_required": "yes"}
            train_items.append((cf, by_attr, truth[cf.attribute]))
    r = ReliabilityMLCR().fit(train_items)
    assert r.acc
    # Off-track confident block must have learned lower accuracy than on-track.
    assert r.acc.get((2, 0), 1.0) < r.acc.get((2, 1), 0.0)

    # Test: on the child conflict, one on-track supporter (a0) vs one
    # off-track higher-confidence supporter (a1); a2 low-confidence (0.6)
    # so it is not part of the conflict but still supports consensus.
    by_attr = {
        "needs_contrast": commits(
            "needs_contrast",
            [("a0", "yes", 0.85), ("a1", "no", 0.9), ("a2", "yes", 0.8)],
        ),
        "renal_screening_required": commits(
            "renal_screening_required",
            [("a0", "yes", 0.8), ("a1", "no", 0.95), ("a2", "yes", 0.6)],
        ),
    }
    cfs = find_conflicts(by_attr, p_min=0.7)
    resolved = r.resolve_session(cfs, by_attr)
    assert resolved["renal_screening_required"] == "yes"
