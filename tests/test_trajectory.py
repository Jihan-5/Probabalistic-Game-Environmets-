"""Tests for the Trajectory record, JSONL round-trip, and the circularity lint."""

import pytest

from pge.trajectory import Trajectory, Turn, assert_non_circular, load_jsonl, save_jsonl


def make_traj(task_id="t0", signals=None):
    return Trajectory(
        task_id=task_id,
        domain="healthcare_v2",
        model="stub",
        seed=42,
        turns=[
            Turn(
                attribute="scan_safe_now",
                guess_value="yes",
                probability=0.8,
                turn=0,
                reasoning="patient had no recent treatment",
                signals=signals if signals is not None else {"noisy_feedback": 1},
            )
        ],
        eval_label=0,
    )


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    trajs = [make_traj(f"t{i}") for i in range(3)]
    assert save_jsonl(trajs, path) == 3
    loaded = load_jsonl(path)
    assert [t.task_id for t in loaded] == ["t0", "t1", "t2"]
    assert loaded[0].turns[0].probability == 0.8
    assert loaded[0].turns[0].signals == {"noisy_feedback": 1}


def test_circular_signal_rejected():
    traj = make_traj(signals={"truth_match": 1})
    with pytest.raises(ValueError, match="blocklist"):
        assert_non_circular(traj)
    with pytest.raises(ValueError):
        traj.validate()


def test_invalid_signal_value_rejected():
    traj = make_traj(signals={"noisy_feedback": 2})
    with pytest.raises(ValueError, match="must be -1, 0, or"):
        traj.validate()


def test_save_validates(tmp_path):
    bad = make_traj(signals={"is_correct": -1})
    with pytest.raises(ValueError):
        save_jsonl([bad], tmp_path / "bad.jsonl")


def test_probability_bounds_checked():
    traj = make_traj()
    traj.turns[0].probability = 1.4
    with pytest.raises(ValueError, match="probability"):
        traj.validate()
