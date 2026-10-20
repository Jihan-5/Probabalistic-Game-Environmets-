"""dbbench v2 — non-circular signals, richer context, reasoning capture.

Differences from v1 (pge/benchmarks/dbbench.py):

1. NON-CIRCULAR SIGNALS. v1's final_answer signal was a match against the
   gold label — the same quantity that defines the eval label (circular,
   see results/RESULTS.md e0_diagnosis). v2's per-turn signals never touch
   the label:
     column_exists     claimed target_column is in the schema
     count_plausible   claimed match_count is an integer in [0, n_rows]
     operation_valid   claimed sql_operation is a known op
     answer_probe      counting tasks: claimed final answer equals the
                       agent's OWN earlier match_count claim
                       (self-consistency); other tasks: the claimed value
                       appears somewhere in the table (existence probe —
                       necessary but not sufficient for correctness)
     resample_agree    majority of k high-temp resamples of the final turn
                       agree with the committed answer
   The gold label is used ONLY for `eval_label`.

2. ACCURACY FIX. v1 saw 13.9% task accuracy (70B), leaving ~15 clean
   training DAGs. v2 (a) filters to counting + single-answer lookup tasks
   with <= 8 columns, (b) shows column types and 3 sample rows in context,
   (c) uses a reasoning-first prompt (pge/prompts/dbbench_v2_system.txt).
   Target: 40-60% accuracy. Verify on a 50-task pilot before scaling.

3. STRUCTURE. Turn dependencies are explicit: match_count and final_answer
   depend on target_column; final_answer also depends on match_count and
   sql_operation. DAG edges follow these dependencies (plus the sequential
   backbone), and each turn's raw reasoning is captured on the Trajectory
   for CoT->DAG extraction and the worked example.
"""

from __future__ import annotations

import re
from typing import Optional

from pge.benchmarks.dbbench import DbbenchTask, _parse_int, _normalize
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory, Turn

DBBENCH_V2_ATTRIBUTES = ["target_column", "match_count", "sql_operation", "final_answer"]

# attribute -> indices of earlier turns it depends on
TURN_DEPENDENCIES = {
    "target_column": [],
    "match_count": [0],
    "sql_operation": [0],
    "final_answer": [0, 1, 2],
}

V2_TASK_TYPES = {
    "counting", "other", "comparison", "ranking",
    "aggregation-SUM", "aggregation-MIN", "aggregation-MAX", "aggregation-AVG",
}
MAX_COLUMNS = 12
AGGREGATION_TYPES = {
    "aggregation-SUM", "aggregation-MIN", "aggregation-MAX", "aggregation-AVG",
}

ATTR_DESCRIPTIONS_V2 = {
    "target_column": (
        "the exact column name whose values are needed to answer the "
        "question. Output one of the column names listed in the context."
    ),
    "match_count": (
        "the number of rows in the table that satisfy the condition implied "
        "by the question (for lookup questions, the number of matching rows; "
        "for counting questions, the count itself). Output a non-negative "
        "integer."
    ),
    "sql_operation": (
        "the primary SQL operation needed (one of: SELECT, WHERE, COUNT, "
        "SUM, MIN, MAX, AVG, ORDER BY). Output one operation."
    ),
    "final_answer": (
        "the exact value that answers the question. Output the answer value "
        "directly (a number, name, date, ...)."
    ),
}


def filter_tasks_v2(tasks: list[DbbenchTask]) -> list[DbbenchTask]:
    out = []
    for t in tasks:
        if t.task_type not in V2_TASK_TYPES:
            continue
        if len(t.columns) > MAX_COLUMNS:
            continue
        if not t.label or not t.rows:
            continue
        out.append(t)
    return out


def setup_answers_v2(task: DbbenchTask, n_sample_rows: int = 3) -> dict:
    """Context shown to the agent: question + schema + sample rows."""
    schema = ", ".join(f"{c} ({ty})" for c, ty in zip(task.columns, task.column_types))
    sample = "; ".join(
        "(" + ", ".join(str(x) for x in row) + ")" for row in task.rows[:n_sample_rows]
    )
    return {
        "question": task.description,
        "table_name": task.table_name,
        "schema": schema,
        "n_rows": len(task.rows),
        "sample_rows": sample,
        "task_type": task.task_type,
    }


# --------------------------------------------------------------- v2 signals


def _table_cells(task: DbbenchTask) -> set[str]:
    return {_normalize(str(x)) for row in task.rows for x in row if x is not None}


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _aggregation_plausible(task: DbbenchTask, value: object) -> int:
    m = _NUM_RE.search(str(value))
    if m is None:
        return -1
    x = float(m.group(0))
    nums = []
    for row in task.rows:
        for cell in row:
            cm = _NUM_RE.fullmatch(str(cell).strip()) if cell is not None else None
            if cm:
                nums.append(float(cm.group(0)))
    if not nums:
        return 1  # nothing to bound against; don't punish
    lo = min(nums)
    hi = sum(n for n in nums if n > 0) if task.task_type == "aggregation-SUM" else max(nums)
    return 1 if lo - 1e-9 <= x <= hi + 1e-9 else -1


def signals_for_turn(
    task: DbbenchTask,
    attribute: str,
    value: object,
    prior_claims: dict[str, object],
    resamples: Optional[list[str]] = None,
) -> dict[str, int]:
    """Detector-visible signals for one commitment. NEVER touches task.label."""
    sig: dict[str, int] = {}
    claim = _normalize(str(value))

    if attribute == "target_column":
        cols = {_normalize(c) for c in task.columns}
        sig["column_exists"] = 1 if claim in cols else -1

    elif attribute == "match_count":
        n = _parse_int(value)
        sig["count_plausible"] = 1 if (n is not None and 0 <= n <= len(task.rows)) else -1

    elif attribute == "sql_operation":
        ops = {
            "select", "where", "count", "sum", "min", "max", "avg",
            "group by", "order by", "distinct", "join",
        }
        sig["operation_valid"] = 1 if any(op in claim for op in ops) else -1

    elif attribute == "final_answer":
        if task.task_type == "counting":
            # Self-consistency: the final count should equal the agent's own
            # match_count claim. Both can be wrong together — that is fine,
            # the probe is about internal coherence, not the label.
            n_ans = _parse_int(value)
            n_claimed = _parse_int(prior_claims.get("match_count", ""))
            if n_ans is None:
                sig["answer_probe"] = -1
            elif n_claimed is not None and n_ans == n_claimed:
                sig["answer_probe"] = 1
            else:
                sig["answer_probe"] = -1
        elif task.task_type in AGGREGATION_TYPES:
            # Numeric-plausibility probe: the claim parses as a number and
            # lies within bounds derivable from the table's numeric cells
            # (min..max for MIN/MAX/AVG, min..sum for SUM). Label untouched.
            sig["answer_probe"] = _aggregation_plausible(task, value)
        else:
            # comparison / ranking / lookup: the claimed value appears
            # somewhere in the table (necessary-but-not-sufficient).
            cells = _table_cells(task)
            hit = claim in cells or any(claim in c or c in claim for c in cells if c)
            sig["answer_probe"] = 1 if hit else -1

        if resamples:
            agree = sum(
                1
                for r in resamples
                if claim and _normalize(_extract_final_value(r)) == claim
            )
            sig["resample_agree"] = 1 if agree * 2 >= len(resamples) else -1

    return sig


_GUESS_LINE_RE = re.compile(
    r"\[[^\]]+\]\s*:\s*(?P<val>[^,\n]+?)\s*,\s*p\s*=", re.IGNORECASE
)


def _extract_final_value(text: str) -> str:
    m = None
    for m in _GUESS_LINE_RE.finditer(text):
        pass
    return m.group("val").strip().strip("\"'") if m else ""


def _split_reasoning(raw: str) -> str:
    """Pull the THOUGHT lines (everything before the commitment line)."""
    lines = [l for l in raw.splitlines() if l.strip()]
    keep = [l for l in lines if not _GUESS_LINE_RE.search(l)]
    return "\n".join(keep).strip()


def aggregate_signal(sig: dict[str, int]) -> int:
    """Collapse named signals into one {-1,0,+1} for DAG node scoring."""
    vals = [v for v in sig.values() if v != 0]
    if not vals:
        return 0
    s = sum(vals)
    return 1 if s > 0 else -1 if s < 0 else 0


# ---------------------------------------------------------------- session


def run_session_v2(
    task: DbbenchTask,
    agent,
    task_idx: int,
    seed: int,
    k_resamples: int = 3,
) -> tuple[Trajectory, ChainOfThoughtDAG]:
    """One full v2 session -> (Trajectory, DAG). Requires an agent exposing
    guess_full() and resample() (see pge/agents/llama_together.py)."""
    setup = setup_answers_v2(task)
    G = ChainOfThoughtDAG()
    turns: list[Turn] = []
    prior_claims: dict[str, object] = {}

    for t, attr in enumerate(DBBENCH_V2_ATTRIBUTES):
        guess, raw = agent.guess_full(t, setup, G)
        samples: list[str] = []
        if attr == "final_answer" and k_resamples > 0:
            samples = agent.resample(t, setup, G, k=k_resamples)

        sig = signals_for_turn(task, attr, guess.value, prior_claims, samples or None)
        prior_claims[attr] = guess.value
        b = aggregate_signal(sig)

        idx = G.add_node(
            GuessNode(
                attribute=attr,
                value=guess.value,
                probability=guess.probability,
                score=bayesian_score(guess.probability, b),
                turn=t,
            )
        )
        for dep in TURN_DEPENDENCIES[attr]:
            if dep < idx:
                G.add_edge(dep, idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)

        turns.append(
            Turn(
                attribute=attr,
                guess_value=str(guess.value),
                probability=guess.probability,
                turn=t,
                raw_response=raw,
                reasoning=_split_reasoning(raw),
                samples=samples,
                signals=sig,
            )
        )

    # EVAL ONLY: label from the gold answer.
    final = _normalize(str(prior_claims.get("final_answer", "")))
    truth = _normalize(task.label)
    correct = bool(final) and (final == truth or final in truth or truth in final)

    traj = Trajectory(
        task_id=f"dbbench_{task_idx}",
        domain="dbbench_v2",
        model=getattr(agent, "model", "unknown"),
        seed=seed,
        turns=turns,
        eval_label=0 if correct else 1,
        label_source="programmatic",
        meta={"task_type": task.task_type, "n_rows": len(task.rows)},
    )
    return traj, G


def dag_from_trajectory(traj: Trajectory) -> ChainOfThoughtDAG:
    """Rebuild the session DAG offline from a cached trajectory (edges are a
    deterministic function of the attribute schedule; scores of the signals).
    Lets all detectors run against JSONL caches without re-running the LLM."""
    G = ChainOfThoughtDAG()
    for turn in traj.turns:
        b = aggregate_signal(turn.signals)
        idx = G.add_node(
            GuessNode(
                attribute=turn.attribute,
                value=turn.guess_value,
                probability=turn.probability,
                score=bayesian_score(turn.probability, b),
                turn=turn.turn,
            )
        )
        deps = TURN_DEPENDENCIES.get(turn.attribute, [])
        for dep in deps:
            if dep < idx:
                G.add_edge(dep, idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)
    return G
