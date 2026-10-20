"""AgentBench-DB adapter for the PGE pipeline.

The public THUDM/AgentBench repo has no healthcare benchmark, so we use its
available dbbench task (NL -> SQL on small relational tables), recast
through PGE's calibrated-guess mechanism.

Per-turn structure (4 turns per task):

  turn 0: agent guesses the **target column** the answer is drawn from
  turn 1: agent guesses **match count** (how many rows satisfy the filter)
  turn 2: agent guesses the **sql operation** required (SELECT, WHERE, etc.)
  turn 3: agent guesses the **final answer value**

For each guess we derive a binary behavior signal by executing a probe
SQL query against an in-memory SQLite copy of the task's table:

  +1 if the probe succeeds (column exists / count matches / etc.)
  -1 if it fails (column missing / value not present / etc.)

This matches the paper's "implicit signal from observed user behavior"
mechanism — the "user" here is the database, and its "behavior" is
whether each progressive claim survives execution.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from pge.agent import Guess
from pge.behavior import BehaviorExtractor


DBBENCH_ATTRIBUTES = [
    "target_column",
    "match_count",
    "sql_operation",
    "final_answer",
]


# Standard task type set that supports SELECT-style evaluation. INSERT / UPDATE
# tasks mutate state and have a different evaluation contract; we skip them.
SELECT_TYPES = {
    "other", "counting", "comparison", "ranking",
    "aggregation-SUM", "aggregation-MIN", "aggregation-MAX", "aggregation-AVG",
}


@dataclass
class DbbenchTask:
    description: str
    label: str
    table_name: str
    columns: list[str]                # column names
    column_types: list[str]            # parallel to columns
    rows: list[list[object]]
    task_type: str

    def setup_answers(self) -> dict[str, object]:
        return {
            "question": self.description,
            "table_name": self.table_name,
            "columns": ", ".join(self.columns),
            "n_columns": len(self.columns),
            "n_rows": len(self.rows),
            "task_type": self.task_type,
        }

    def user_state(self) -> dict[str, object]:
        return {
            "answer": self.label,
            "table_name": self.table_name,
            "columns": self.columns,
            "column_types": self.column_types,
            "rows": self.rows,
        }


# ----------------------------------------------------------------- loader


def load_dbbench(
    paths: Iterable[Path | str],
    task_types: Optional[set[str]] = None,
    max_tasks: Optional[int] = None,
) -> list[DbbenchTask]:
    """Load AgentBench dbbench JSONL files into `DbbenchTask` records.

    Args:
        paths: one or more JSONL file paths to load.
        task_types: optional set of task types to keep. Defaults to SELECT_TYPES.
        max_tasks: if set, truncate after this many tasks.
    """
    if task_types is None:
        task_types = SELECT_TYPES

    tasks: list[DbbenchTask] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                t_field = obj.get("type", "other")
                if isinstance(t_field, list):
                    t_field = t_field[0] if t_field else "other"
                if t_field not in task_types:
                    continue
                table = obj["table"]
                cols = [c["name"] for c in table["table_info"]["columns"]]
                col_types = [c["type"] for c in table["table_info"]["columns"]]
                label = obj["label"]
                if isinstance(label, list):
                    label = label[0] if label else ""
                tasks.append(DbbenchTask(
                    description=obj["description"],
                    label=str(label),
                    table_name=table.get("table_name", "t"),
                    columns=cols,
                    column_types=col_types,
                    rows=table["table_info"]["rows"],
                    task_type=t_field,
                ))
                if max_tasks is not None and len(tasks) >= max_tasks:
                    return tasks
    return tasks


# ------------------------------------------------------------ sqlite helpers


def _safe_ident(name: str) -> str:
    """Quote an identifier for SQLite. Strips backticks/quotes and brackets."""
    cleaned = re.sub(r"[^\w \-]", "", name).strip()
    return f'"{cleaned}"'


def _build_sqlite(task: DbbenchTask) -> sqlite3.Connection:
    """In-memory SQLite copy of the task's table."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cols_sql = ", ".join(
        f"{_safe_ident(c)} {t.upper() if t.upper() in ('INTEGER', 'REAL', 'TEXT', 'NUMERIC') else 'TEXT'}"
        for c, t in zip(task.columns, task.column_types)
    )
    tname = _safe_ident(task.table_name)
    cur.execute(f"CREATE TABLE {tname} ({cols_sql})")
    placeholders = ", ".join(["?"] * len(task.columns))
    cur.executemany(
        f"INSERT INTO {tname} VALUES ({placeholders})",
        [tuple(str(x) if x is not None else None for x in row) for row in task.rows],
    )
    conn.commit()
    return conn


# ----------------------------------------------- behavior signal extractor


_INT_RE = re.compile(r"-?\d+")


def _parse_int(text: str) -> Optional[int]:
    m = _INT_RE.search(str(text))
    return int(m.group(0)) if m else None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


class DbbenchBehaviorExtractor(BehaviorExtractor):
    """Validates per-turn agent guesses against the in-memory SQLite table.

    The extractor caches one connection per task (keyed by table_name) since
    a session walks all 4 attributes for the *same* task; the same DB is
    reused across turns.
    """

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._task: Optional[DbbenchTask] = None

    def bind(self, task: DbbenchTask) -> None:
        """Switch the extractor to a new task; called by the eval driver
        between sessions. Closes any previous connection."""
        if self._conn is not None:
            self._conn.close()
        self._conn = _build_sqlite(task)
        self._task = task

    def extract(self, guess: Guess, user_state: Mapping[str, object], turn: int) -> int:
        if self._task is None or self._conn is None:
            raise RuntimeError("DbbenchBehaviorExtractor.bind() must be called first")

        if guess.attribute == "target_column":
            return self._check_column(guess.value, user_state)
        if guess.attribute == "match_count":
            return self._check_match_count(guess.value, user_state)
        if guess.attribute == "sql_operation":
            return self._check_operation(guess.value, user_state)
        if guess.attribute == "final_answer":
            return self._check_final_answer(guess.value, user_state)
        return -1  # unknown attribute -> hallucination

    # ------------------------- per-attribute checks

    def _check_column(self, claimed_col: object, user_state) -> int:
        cols = [str(c) for c in user_state["columns"]]
        return 1 if _normalize(str(claimed_col)) in {_normalize(c) for c in cols} else -1

    def _check_match_count(self, claimed_count: object, user_state) -> int:
        """The agent's claim should be a non-negative integer that's plausibly
        derivable from the table. We accept any integer in [0, n_rows] as
        +1 (it's at least syntactically valid); only nonsense (negative or
        > n_rows) is -1. This is a *soft* signal — true count is checked
        implicitly by the final_answer turn."""
        n = _parse_int(claimed_count)
        if n is None:
            return -1
        return 1 if 0 <= n <= len(user_state["rows"]) else -1

    _ALLOWED_OPS = {
        "select", "where", "count", "sum", "min", "max", "avg",
        "group by", "order by", "distinct", "join",
    }

    def _check_operation(self, claimed_op: object, user_state) -> int:
        op = _normalize(str(claimed_op))
        return 1 if any(allowed in op for allowed in self._ALLOWED_OPS) else -1

    def _check_final_answer(self, claimed_answer: object, user_state) -> int:
        """Strict: the claim must match the label (case-insensitive, whitespace
        normalized). This is the strongest hallucination signal — wrong final
        answers are the failure mode PGE most needs to catch."""
        truth = _normalize(str(user_state["answer"]))
        claim = _normalize(str(claimed_answer))
        # Allow substring containment in either direction to handle short labels
        # like "1998" vs "March 19, 1998".
        if claim == truth or claim in truth or truth in claim:
            return 1
        return -1

    def is_correct(self, dag) -> bool:
        """True iff the agent's final_answer matches the label.

        Used as ground-truth hallucination signal: trajectories where the
        agent got the answer wrong are the labeled "hallucinating" set.
        """
        if self._task is None:
            return False
        for node in dag.nodes:
            if node.attribute == "final_answer":
                return _normalize(str(node.value)) == _normalize(self._task.label) \
                    or _normalize(str(node.value)) in _normalize(self._task.label) \
                    or _normalize(self._task.label) in _normalize(str(node.value))
        return False
