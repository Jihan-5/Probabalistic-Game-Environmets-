"""Tests for dbbench v2: non-circular signals, session flow, DAG rebuild."""

import pytest

from pge.benchmarks.dbbench import DbbenchTask
from pge.benchmarks.dbbench_v2 import (
    dag_from_trajectory,
    filter_tasks_v2,
    run_session_v2,
    signals_for_turn,
)
from pge.trajectory import CIRCULAR_SIGNAL_NAMES


def make_task(task_type="counting", label="2"):
    return DbbenchTask(
        description="How many players are from Spain?",
        label=label,
        table_name="players",
        columns=["name", "country", "goals"],
        column_types=["text", "text", "integer"],
        rows=[["Ana", "Spain", "3"], ["Bo", "France", "1"], ["Cruz", "Spain", "5"]],
        task_type=task_type,
    )


class FakeAgent:
    """Deterministic agent stub with the guess_full/resample interface."""

    model = "fake"

    def __init__(self, answers):
        self.answers = answers  # attr -> (value, p)

    def guess_full(self, turn, setup, history):
        from pge.agent import Guess
        from pge.benchmarks.dbbench_v2 import DBBENCH_V2_ATTRIBUTES

        attr = DBBENCH_V2_ATTRIBUTES[turn]
        value, p = self.answers[attr]
        raw = f"THOUGHT: thinking about {attr}\n[{attr}]: {value}, p={p}"
        return Guess(attribute=attr, value=value, probability=p), raw

    def resample(self, turn, setup, history, k=3, temperature=0.9):
        attr = "final_answer"
        value, p = self.answers[attr]
        return [f"[{attr}]: {value}, p={p}"] * k


GOOD = {
    "target_column": ("country", 0.9),
    "match_count": ("2", 0.8),
    "sql_operation": ("COUNT", 0.9),
    "final_answer": ("2", 0.85),
}
BAD = {
    "target_column": ("nationality", 0.9),  # column doesn't exist
    "match_count": ("7", 0.8),              # > n_rows
    "sql_operation": ("COUNT", 0.9),
    "final_answer": ("5", 0.9),             # inconsistent with match_count claim
}


def test_signals_never_use_blocked_names():
    task = make_task()
    for attr, (value, _) in GOOD.items():
        sig = signals_for_turn(task, attr, value, dict(GOOD))
        assert not (set(sig) & CIRCULAR_SIGNAL_NAMES)


def test_good_session_positive_signals_and_clean_label():
    traj, dag = run_session_v2(make_task(), FakeAgent(GOOD), 0, seed=1)
    assert traj.eval_label == 0
    all_sigs = {k: v for t in traj.turns for k, v in t.signals.items()}
    assert all_sigs["column_exists"] == 1
    assert all_sigs["count_plausible"] == 1
    assert all_sigs["answer_probe"] == 1  # 2 == claimed match_count
    assert all_sigs["resample_agree"] == 1
    traj.validate()  # non-circularity lint passes


def test_bad_session_negative_signals_and_hallucination_label():
    traj, dag = run_session_v2(make_task(), FakeAgent(BAD), 0, seed=1)
    assert traj.eval_label == 1
    all_sigs = {k: v for t in traj.turns for k, v in t.signals.items()}
    assert all_sigs["column_exists"] == -1
    assert all_sigs["count_plausible"] == -1
    assert all_sigs["answer_probe"] == -1


def test_signals_are_not_label_functions():
    # Same claims, different gold label: signals must be identical.
    t1, t2 = make_task(label="2"), make_task(label="999")
    for attr, (value, _) in GOOD.items():
        s1 = signals_for_turn(t1, attr, value, dict(GOOD))
        s2 = signals_for_turn(t2, attr, value, dict(GOOD))
        assert s1 == s2
    # ...but the eval label may differ (it is allowed to use the gold label).
    a, _ = run_session_v2(t1, FakeAgent(GOOD), 0, seed=1)
    b, _ = run_session_v2(t2, FakeAgent(GOOD), 0, seed=1)
    assert a.eval_label == 0 and b.eval_label == 1


def test_existence_probe_for_lookup_tasks():
    task = make_task(task_type="other", label="Ana")
    claims = dict(GOOD, final_answer=("Ana", 0.9))
    sig = signals_for_turn(task, "final_answer", "Ana", claims)
    assert sig["answer_probe"] == 1  # value exists in the table
    sig = signals_for_turn(task, "final_answer", "Zed", claims)
    assert sig["answer_probe"] == -1


def test_dag_structure_and_offline_rebuild():
    traj, dag = run_session_v2(make_task(), FakeAgent(GOOD), 0, seed=1)
    assert len(dag) == 4
    assert (0, 3) in dag.edges and (1, 3) in dag.edges and (2, 3) in dag.edges
    rebuilt = dag_from_trajectory(traj)
    assert rebuilt.edges == dag.edges
    assert [n.probability for n in rebuilt.nodes] == [n.probability for n in dag.nodes]


def test_filter_keeps_small_select_tasks():
    keep = make_task()
    drop_type = make_task(task_type="INSERT")
    drop_cols = make_task()
    drop_cols.columns = [f"c{i}" for i in range(13)]
    drop_cols.column_types = ["text"] * 13
    out = filter_tasks_v2([keep, drop_type, drop_cols])
    assert out == [keep]
