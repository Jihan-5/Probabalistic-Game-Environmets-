"""Semantic-entropy baseline (Kuhn et al. 2023, adapted to trajectories).

Failure mode targeted: hallucinating agents are unstable under resampling —
high-temperature resamples of the same turn scatter across semantically
distinct answers, while grounded agents cluster on one. Score = Shannon
entropy over answer clusters formed from the final turn's committed guess
plus its k resamples (0 if fewer than 2 answers are available).

Clustering is exact-match by default (whitespace/case/punctuation
normalized, numeric strings compared as floats). `cluster="llm"` upgrades
to the paper's bidirectional-entailment clustering via an injectable
`entail_fn` — no API dependency in this module.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

from pge.baselines.base import TrajectoryScorer
from pge.trajectory import Trajectory, Turn

# Commitment-line format shared across benchmarks: "[attr]: value, p=0.8".
_GUESS_LINE_RE = re.compile(
    r"\[[^\]]+\]\s*:\s*(?P<val>[^,\n]+?)\s*,\s*p\s*=", re.IGNORECASE
)

_EDGE_PUNCT = " \t\"'.,;:!?()[]{}"


def _extract_final_value(text: str) -> str:
    """Last committed value in a resample text; '' if none found."""
    m = None
    for m in _GUESS_LINE_RE.finditer(text):
        pass
    return m.group("val").strip().strip("\"'") if m else ""


def _normalize(value: str) -> str:
    """Canonical form for answer comparison: case/whitespace/edge-punctuation
    insensitive; numeric strings collapse to a float representation so that
    "2" == "2.0"."""
    v = value.strip().strip(_EDGE_PUNCT).lower()
    v = re.sub(r"\s+", " ", v)
    try:
        return repr(float(v))
    except ValueError:
        return v


def _cluster_sizes_exact(answers: list[str]) -> list[int]:
    counts: dict[str, int] = {}
    for a in answers:
        key = _normalize(a)
        counts[key] = counts.get(key, 0) + 1
    return list(counts.values())


def _cluster_sizes_entail(
    answers: list[str], entail_fn: Callable[[str, str], bool]
) -> list[int]:
    """Greedy bidirectional-entailment clustering against each cluster's
    first member (representative), as in Kuhn et al."""
    reps: list[str] = []
    sizes: list[int] = []
    for a in answers:
        for i, r in enumerate(reps):
            if entail_fn(a, r) and entail_fn(r, a):
                sizes[i] += 1
                break
        else:
            reps.append(a)
            sizes.append(1)
    return sizes


class SemanticEntropy(TrajectoryScorer):
    name = "semantic_entropy"

    def __init__(
        self,
        cluster: str = "exact",
        entail_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        if cluster not in ("exact", "llm"):
            raise ValueError(f"unknown cluster mode: {cluster!r}")
        if cluster == "llm" and entail_fn is None:
            raise ValueError("cluster='llm' requires an entail_fn")
        self.cluster = cluster
        self.entail_fn = entail_fn

    # ------------------------------------------------------------- internals

    def _turn_answers(self, turn: Turn) -> list[str]:
        answers = [turn.guess_value]
        for s in turn.samples:
            v = _extract_final_value(s)
            if v:
                answers.append(v)
        return answers

    def _turn_entropy(self, turn: Turn) -> float:
        answers = self._turn_answers(turn)
        if len(answers) < 2:
            return 0.0
        if self.cluster == "llm":
            sizes = _cluster_sizes_entail(answers, self.entail_fn)
        else:
            sizes = _cluster_sizes_exact(answers)
        total = sum(sizes)
        return -sum((c / total) * math.log(c / total) for c in sizes if c > 0)

    # ------------------------------------------------------------- interface

    def score(self, trajectory: Trajectory, dag=None) -> float:
        if not trajectory.turns:
            return 0.0
        return float(self._turn_entropy(trajectory.turns[-1]))


class MeanTurnEntropy(SemanticEntropy):
    """Mean semantic entropy across all turns that carry resamples — catches
    mid-trajectory instability the final-turn score misses. 0 if no turn
    has samples."""

    name = "mean_turn_entropy"

    def score(self, trajectory: Trajectory, dag=None) -> float:
        ents = [
            self._turn_entropy(t) for t in trajectory.turns if t.samples
        ]
        if not ents:
            return 0.0
        return float(sum(ents) / len(ents))
