"""Constraint interface for MLCR (paper Section 3.4).

Omega_C is the set of operation assignments satisfying all hard constraints
(medical protocols, logical consistency, etc.). The MLCR optimizer queries
the constraint set both for per-node feasible operations (M-step search space)
and to validate full assignments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Constraints(ABC):
    """Hard-constraint set Omega_C for MLCR."""

    @abstractmethod
    def feasible_ops(self, node_idx: int, current_assignment: Sequence[object]) -> list[object]:
        """Operations available for node `node_idx` given the current full assignment."""

    @abstractmethod
    def is_consistent(self, assignment: Sequence[object]) -> bool:
        """True iff `assignment` (one operation per node) satisfies all hard constraints."""


class SimpleConstraints(Constraints):
    """Per-node candidate set with optional pairwise forbidden pairs.

    Useful both as a default for the demo and as a sanity-check fixture for tests.

    Parameters
    ----------
    candidates : list[list[object]]
        For each node, the list of allowed operations.
    forbidden_pairs : set[tuple[int, object, int, object]]
        Set of (i, op_i, j, op_j) tuples that must NOT both appear together.
    """

    def __init__(
        self,
        candidates: list[list[object]],
        forbidden_pairs: set | None = None,
    ):
        self.candidates = candidates
        self.forbidden_pairs = forbidden_pairs or set()

    def feasible_ops(self, node_idx, current_assignment):
        allowed = []
        for op in self.candidates[node_idx]:
            ok = True
            for j, other in enumerate(current_assignment):
                if j == node_idx:
                    continue
                if (node_idx, op, j, other) in self.forbidden_pairs:
                    ok = False
                    break
                if (j, other, node_idx, op) in self.forbidden_pairs:
                    ok = False
                    break
            if ok:
                allowed.append(op)
        return allowed or list(self.candidates[node_idx])  # fall back rather than empty

    def is_consistent(self, assignment):
        for i, op_i in enumerate(assignment):
            if op_i not in self.candidates[i]:
                return False
            for j, op_j in enumerate(assignment):
                if i == j:
                    continue
                if (i, op_i, j, op_j) in self.forbidden_pairs:
                    return False
        return True
