"""Tests for pge/baselines/posterior.py."""

from __future__ import annotations

import copy
import itertools

import numpy as np

from eval.bootstrap_ci import auroc
from pge.baselines.posterior import (
    EvidenceNB,
    IndependentPosterior,
    label_threshold,
    poisson_binomial_tail,
)
from pge.benchmarks.healthcare_v2 import generate_dataset_v2
from pge.trajectory import Trajectory, Turn


def test_poisson_binomial_tail_matches_enumeration():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.05, 0.95, size=7)
    for k in range(len(q) + 1):
        brute = sum(
            np.prod([qi if w else 1 - qi for qi, w in zip(q, ws)])
            for ws in itertools.product((0, 1), repeat=len(q))
            if sum(ws) >= k
        )
        assert abs(poisson_binomial_tail(q, k) - brute) < 1e-12


def test_label_threshold_matches_eval_label_rule():
    # eval_label_v2: wrong / n_high >= 0.5
    for n in range(1, 13):
        k = label_threshold(n)
        assert k / n >= 0.5 and (k - 1) / n < 0.5


def _synthetic(n=600, seed=7):
    trajs, dags = generate_dataset_v2(n, seed=seed)
    y = np.array([t.eval_label for t in trajs])
    return trajs, dags, y


def test_posterior_detects_on_synthetic_healthcare():
    trajs, dags, y = _synthetic()
    tr, te = slice(0, 420), slice(420, None)
    for det in (IndependentPosterior(), IndependentPosterior(by_value=True)):
        det.fit(trajs[tr], dags[tr], y[tr])
        s = np.array([det.score(t) for t in trajs[te]])
        assert np.all((s >= 0) & (s <= 1))
        assert auroc(s, y[te]) > 0.6
        assert 0.05 <= det.eps <= 0.45


def test_contradiction_raises_posterior():
    trajs, dags, y = _synthetic(300)
    det = IndependentPosterior(by_value=True).fit(trajs, dags, y)
    for traj in trajs[:50]:
        for i, turn in enumerate(traj.turns):
            if turn.probability >= 0.6 and turn.signals.get("noisy_feedback") == 1:
                worse = copy.deepcopy(traj)
                worse.turns[i].signals["noisy_feedback"] = -1
                assert det.score(worse) >= det.score(traj) - 1e-12
                break


def test_score_ignores_eval_label():
    trajs, dags, y = _synthetic(300)
    det = IndependentPosterior(by_value=True, use_samples=True).fit(trajs, dags, y)
    for traj in trajs[:30]:
        flipped = copy.deepcopy(traj)
        flipped.eval_label = 1 - traj.eval_label
        assert det.score(flipped) == det.score(traj)


def _db_traj(i, good: bool):
    probe = 1 if good else -1
    turns = [
        Turn("target_column", "c", 0.9, 0, signals={"column_exists": 1}),
        Turn("match_count", "3", 0.8, 1, signals={"count_plausible": 1}),
        Turn("sql_operation", "count", 0.8, 2, signals={"operation_valid": 1}),
        Turn(
            "final_answer",
            "3",
            0.8 if good else 0.6,
            3,
            samples=["[final_answer]: 3, p=0.8"] * (3 if good else 1)
            + ["[final_answer]: 5, p=0.5"] * (0 if good else 2),
            signals={"answer_probe": probe},
        ),
    ]
    return Trajectory(f"db_{i}", "dbbench_v2", "m", 0, turns, eval_label=0 if good else 1)


def test_evidence_nb_separates_probe_evidence():
    trajs = [_db_traj(i, good=i % 2 == 0) for i in range(40)]
    y = np.array([t.eval_label for t in trajs])
    det = EvidenceNB().fit(trajs, None, y)
    s = np.array([det.score(t) for t in trajs])
    assert auroc(s, y) == 1.0
