"""Tests for the offline detector-evaluation harness (eval.score_trajectories).

All fixtures are SYNTHETIC in-memory trajectories shaped like the dbbench_v2
schedule (target_column, match_count, sql_operation, final_answer with
non-circular signals) and saved via save_jsonl to tmp_path — zero LLM cost,
no dependence on the optional concurrent baseline modules (--core-only path).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.score_trajectories import (
    average_precision,
    build_dag,
    ece_10bin,
    run_experiment,
    split_task_ids,
)
from pge.trajectory import Trajectory, Turn, save_jsonl

DBBENCH_SCHEDULE = [
    ("target_column", "column_exists"),
    ("match_count", "count_plausible"),
    ("sql_operation", "operation_valid"),
    ("final_answer", "answer_probe"),
]


def _make_traj(task_idx: int, hallucinating: bool, rng: np.random.Generator,
               seed: int = 42, model: str = "synthetic") -> Trajectory:
    """dbbench_v2-shaped synthetic trajectory. Hallucinating traces carry
    confident-contradicted turns; clean traces confident-supported ones."""
    turns = []
    for t, (attr, sig_name) in enumerate(DBBENCH_SCHEDULE):
        if hallucinating:
            signal = -1 if rng.random() < 0.8 else 1
            prob = float(np.clip(rng.normal(0.85, 0.08), 0.05, 0.99))
        else:
            signal = 1 if rng.random() < 0.85 else -1
            prob = float(np.clip(rng.normal(0.75, 0.10), 0.05, 0.99))
        turns.append(
            Turn(
                attribute=attr,
                guess_value=f"v{task_idx}_{t}",
                probability=prob,
                turn=t,
                signals={sig_name: signal},
            )
        )
    return Trajectory(
        task_id=f"dbbench_{task_idx}",
        domain="dbbench_v2",
        model=model,
        seed=seed,
        turns=turns,
        eval_label=1 if hallucinating else 0,
    )


def _make_cache(path, n_tasks: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    trajs = [
        _make_traj(i, hallucinating=(i % 2 == 1), rng=rng) for i in range(n_tasks)
    ]
    save_jsonl(trajs, path)
    return trajs


# --------------------------------------------------------------- splitting


def test_split_deterministic_and_disjoint():
    ids = [f"dbbench_{i}" for i in range(50)]
    tr1, te1 = split_task_ids(ids, train_frac=0.7, seed=42)
    tr2, te2 = split_task_ids(ids, train_frac=0.7, seed=42)
    assert tr1 == tr2 and te1 == te2  # deterministic
    assert tr1 & te1 == set()  # disjoint
    assert tr1 | te1 == set(ids)  # complete
    assert len(tr1) == 35  # int(round(0.7 * 50))

    # A different seed reshuffles the assignment.
    tr3, _ = split_task_ids(ids, train_frac=0.7, seed=123)
    assert tr3 != tr1


def test_split_is_by_task_id_not_row():
    # The same task_id repeated across "files" (model/seed variants) must land
    # entirely on one side of the split.
    ids = [f"dbbench_{i}" for i in range(20)] * 3  # 3 rows per task
    train, test = split_task_ids(ids, train_frac=0.7, seed=7)
    assert train & test == set()
    assert len(train) + len(test) == 20  # unique tasks, not rows


# ------------------------------------------------------------------ metrics


def test_auprc_perfect_ranking_is_one():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0])
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_auprc_worst_ranking():
    # All positives ranked below all negatives: AP = mean of k/(n_neg+k).
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([1, 1, 0, 0])
    expected = 0.5 * (1 / 3 + 2 / 4)
    assert average_precision(scores, labels) == pytest.approx(expected)


def test_auprc_degenerate_labels_nan():
    assert np.isnan(average_precision(np.array([0.5, 0.4]), np.array([1, 1])))
    assert np.isnan(average_precision(np.array([0.5, 0.4]), np.array([0, 0])))


def test_ece_calibrated_and_miscalibrated():
    # Perfectly calibrated: every prob-0.7 bin has 70% positives.
    probs = np.array([0.7] * 10)
    outcomes = np.array([1] * 7 + [0] * 3)
    assert ece_10bin(probs, outcomes) == pytest.approx(0.0)

    # Maximally overconfident: p=0.95 but never right -> ECE 0.95.
    probs = np.array([0.95] * 8)
    outcomes = np.zeros(8, dtype=int)
    assert ece_10bin(probs, outcomes) == pytest.approx(0.95)

    assert np.isnan(ece_10bin(np.array([]), np.array([])))


# -------------------------------------------------------------- end-to-end


def test_scorer_end_to_end(tmp_path):
    cache = tmp_path / "synthetic_42.jsonl"
    _make_cache(cache, n_tasks=40)
    out_root = tmp_path / "results"

    report = run_experiment(
        exp="test_exp",
        paths=[str(cache)],
        seeds=[42],
        train_frac=0.7,
        blocks=6,
        n_boot=50,  # keep the bootstrap cheap
        core_only=True,  # never require the concurrent baseline modules
        out_root=out_root,
    )

    # Summary written and populated.
    summary = out_root / "test_exp" / "summary.txt"
    assert summary.exists()
    text = summary.read_text()
    assert "AUROC" in text and "AUPRC" in text and "calibration" in text
    assert report.strip() in text

    # npz per seed with labels, task_ids, and detector scores.
    npz_path = out_root / "test_exp" / "seed42.npz"
    assert npz_path.exists()
    with np.load(npz_path) as z:
        score_keys = [k for k in z.files if k.startswith("score_")]
        assert score_keys, "no detector scores saved"
        n_test = len(z["labels"])
        assert n_test == 40 - int(round(0.7 * 40))
        assert len(z["task_ids"]) == n_test
        assert z["task_ids"].dtype.kind in ("U", "S")
        for k in score_keys:
            assert len(z[k]) == n_test
        # Core zoo must include graphon variants and the supervised arm.
        assert any(k.startswith("score_graphon_") for k in score_keys)
        assert "score_dag_logreg" in score_keys
        # Calibration report fields present.
        assert float(z["ece_before"]) >= 0.0
        assert float(z["temperature"]) > 0.0

    # Signals are strongly separating by construction: the best detector
    # should be far above chance on the test split.
    with np.load(npz_path) as z:
        from eval.bootstrap_ci import auroc

        best = max(
            auroc(z[k], z["labels"]) for k in z.files if k.startswith("score_")
        )
        assert best > 0.8


def test_dag_rebuild_shape():
    rng = np.random.default_rng(1)
    traj = _make_traj(0, hallucinating=True, rng=rng)
    dag = build_dag(traj)
    assert len(dag) == 4
    # dbbench_v2 dependency edges + sequential backbone.
    assert (0, 1) in dag.edges and (0, 3) in dag.edges and (2, 3) in dag.edges
