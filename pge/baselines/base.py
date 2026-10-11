"""Common detector interface for the rebuild.

Every hallucination detector — baseline or PGE variant — implements
`TrajectoryScorer`: fit on training data (no-op for zero-shot methods),
then emit a scalar per trajectory where HIGHER = MORE LIKELY HALLUCINATED.
Detectors receive the trajectory and its (already-built) reasoning DAG so
graph-based and text-based methods share one evaluation harness. Detector
code must only read detector-visible fields (turns, signals, DAG) — never
`eval_label` (enforced by review + the non-circularity lint on signals).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pge.dag import ChainOfThoughtDAG
from pge.trajectory import Trajectory


class TrajectoryScorer(ABC):
    name: str = "unnamed"

    def fit(
        self,
        trajectories: Sequence[Trajectory],
        dags: Optional[Sequence[ChainOfThoughtDAG]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> "TrajectoryScorer":
        """Train on labeled data. Default: no-op (zero-shot detectors).

        `labels` are the training-split eval labels — legitimate supervision
        for supervised detectors; never available at score time.
        """
        return self

    @abstractmethod
    def score(
        self, trajectory: Trajectory, dag: Optional[ChainOfThoughtDAG] = None
    ) -> float:
        """Higher = more likely hallucinated."""
