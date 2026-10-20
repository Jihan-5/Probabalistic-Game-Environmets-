"""Chain-of-thought DAG (paper Section 3.3).

Each node is a scored guess; edges connect sequential guesses to capture
reasoning dependencies. Light-weight (no NetworkX dep) — adjacency matrix
is materialized on demand for graphon ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GuessNode:
    attribute: str
    value: object
    probability: float
    score: float = 0.0
    turn: int = 0
    operation: Optional[object] = None  # used by MLCR
    # Raw feedback signal for this commitment (+1 / -1 / 0), when known. Only
    # used to DISPLAY feedback to agents (pge.agents.paid_client); detectors
    # read Trajectory signals, never this field.
    signal: Optional[int] = None


@dataclass
class ChainOfThoughtDAG:
    nodes: list[GuessNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)  # (parent_idx, child_idx)

    def add_node(self, node: GuessNode) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_edge(self, parent_idx: int, child_idx: int) -> None:
        if parent_idx == child_idx:
            raise ValueError("self-loops not allowed")
        if parent_idx >= child_idx:
            raise ValueError(
                "edges must go from earlier to later node (DAG topological order)"
            )
        if not (0 <= parent_idx < len(self.nodes) and 0 <= child_idx < len(self.nodes)):
            raise IndexError("edge endpoints out of range")
        self.edges.append((parent_idx, child_idx))

    def __len__(self) -> int:
        return len(self.nodes)

    def adjacency(self) -> np.ndarray:
        """Return n x n {0,1} adjacency matrix (upper-triangular, since DAG)."""
        n = len(self.nodes)
        A = np.zeros((n, n), dtype=np.int8)
        for p, c in self.edges:
            A[p, c] = 1
        return A

    def scores(self) -> np.ndarray:
        return np.array([n.score for n in self.nodes], dtype=float)

    def probabilities(self) -> np.ndarray:
        return np.array([n.probability for n in self.nodes], dtype=float)
