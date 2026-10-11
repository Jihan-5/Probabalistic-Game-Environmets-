"""Unified trajectory record: the single artifact every detector consumes.

Design rule (non-negotiable for the rebuild): **evaluation labels may be
computed from ground truth; detector inputs may not be deterministic
functions of that same ground truth.** `eval_label` lives on the trajectory
for scoring/evaluation only. Detector-visible information is confined to
`turns[*]` (prompts, responses, reasoning, parsed guesses, resamples) and
`turns[*].signals` — named behavioral probe signals in {-1, 0, +1} that must
come from a non-circular channel (self-consistency probes, noisy/partial
user feedback, execution success, resample agreement, ...).

`assert_non_circular` mechanically rejects signal names that are known
label-derived probes from the old pipeline, so the circular-evaluation bug
documented in results/RESULTS.md (e0_diagnosis) cannot silently return.

Trajectories are generated once per (domain, model, seed) — the only paid
step — and cached as JSONL under data/trajectories/; all detectors and
ablations run offline against the cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# Signal names from the pre-rebuild pipeline that were direct functions of the
# eval label (truth-matching). Any trajectory carrying one of these fails the
# lint. Extend this list whenever a new label-derived probe is identified.
CIRCULAR_SIGNAL_NAMES = frozenset(
    {
        "truth_match",
        "label_match",
        "is_correct",
        "gold_match",
        "ground_truth_match",
    }
)

_VALID_SIGNAL_VALUES = (-1, 0, 1)


@dataclass
class Turn:
    """One interaction turn: the agent's commitment plus observable evidence."""

    attribute: str
    guess_value: str
    probability: float
    turn: int
    prompt: str = ""
    raw_response: str = ""
    reasoning: str = ""
    samples: list[str] = field(default_factory=list)  # k resamples (entropy/SelfCheck)
    signals: dict[str, int] = field(default_factory=dict)  # name -> {-1, 0, +1}


@dataclass
class Trajectory:
    task_id: str
    domain: str
    model: str
    seed: int
    turns: list[Turn] = field(default_factory=list)
    eval_label: int = 0  # 1 = hallucinating. GROUND-TRUTH-DERIVED; EVAL ONLY.
    label_source: str = "programmatic"  # programmatic | llm_judge | human
    meta: dict = field(default_factory=dict)  # cost, tokens, config hash, ...

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        assert_non_circular(self)
        for t in self.turns:
            if not 0.0 <= t.probability <= 1.0:
                raise ValueError(
                    f"{self.task_id} turn {t.turn}: probability {t.probability} "
                    "outside [0, 1]"
                )
        if self.eval_label not in (0, 1):
            raise ValueError(f"{self.task_id}: eval_label must be 0 or 1")

    # --------------------------------------------------------------- IO

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        turns = [Turn(**t) for t in d.get("turns", [])]
        return cls(
            task_id=d["task_id"],
            domain=d["domain"],
            model=d["model"],
            seed=int(d["seed"]),
            turns=turns,
            eval_label=int(d.get("eval_label", 0)),
            label_source=d.get("label_source", "programmatic"),
            meta=d.get("meta", {}),
        )


def assert_non_circular(traj: Trajectory) -> None:
    """Fail loudly if any detector-visible signal is a known label probe."""
    for t in traj.turns:
        for name, value in t.signals.items():
            if name in CIRCULAR_SIGNAL_NAMES:
                raise ValueError(
                    f"{traj.task_id} turn {t.turn}: signal '{name}' is on the "
                    "circular-signal blocklist (deterministic function of the "
                    "eval label). Use a non-circular probe channel instead."
                )
            if value not in _VALID_SIGNAL_VALUES:
                raise ValueError(
                    f"{traj.task_id} turn {t.turn}: signal '{name}'={value} "
                    "must be -1, 0, or +1"
                )


# ------------------------------------------------------------------- JSONL


def save_jsonl(trajectories: Iterable[Trajectory], path: str | Path) -> int:
    """Validate and write trajectories; returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for traj in trajectories:
            traj.validate()
            f.write(json.dumps(traj.to_dict()) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[Trajectory]:
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = Trajectory.from_dict(json.loads(line))
            traj.validate()
            out.append(traj)
    return out


def iter_jsonl(path: str | Path) -> Iterator[Trajectory]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                traj = Trajectory.from_dict(json.loads(line))
                traj.validate()
                yield traj
