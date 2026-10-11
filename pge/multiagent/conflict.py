"""Conflict detection over multi-agent commitments.

A conflict is an attribute where at least two agents commit to DIFFERENT
values, each with reported probability >= p_min. Low-confidence disagreement
is not a conflict (nobody is standing behind their value), and confident
agreement is not a conflict either.
"""

from __future__ import annotations

from dataclasses import dataclass

from pge.multiagent.commitments import Commitment


@dataclass
class Conflict:
    """A conflicted attribute plus ALL commitments for it (not just the
    confident ones) — resolvers may want the full vote."""

    attribute: str
    commitments: list[Commitment]


def find_conflicts(
    commitments_by_attr: dict[str, list[Commitment]],
    p_min: float = 0.7,
) -> list[Conflict]:
    """Return conflicts in schedule (dict insertion) order."""
    conflicts: list[Conflict] = []
    for attribute, commitments in commitments_by_attr.items():
        confident_values = {
            c.value for c in commitments if c.probability >= p_min
        }
        if len(confident_values) >= 2:
            conflicts.append(Conflict(attribute, list(commitments)))
    return conflicts
