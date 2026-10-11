"""Multi-agent commitment collection over the healthcare-v2 environment.

Each agent profile independently walks a session's schedule with its own
`prior_guesses` dict, so error cascades are per-agent (agents do not see each
other's guesses). Profiles control the hallucinator flag and a Gaussian
probability jitter (`noise`), giving heterogeneous agent populations for free
(no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pge.benchmarks.healthcare_v2 import AgentGuessV2, SessionV2, synthetic_agent_guess


@dataclass
class Commitment:
    """One agent's committed value for one attribute, with reported confidence."""

    agent_id: str
    attribute: str
    value: str
    probability: float


def collect_commitments(
    session: SessionV2,
    agent_profiles: list[dict],
    rng: np.random.Generator,
) -> dict[str, list[Commitment]]:
    """Run every agent profile through `session.schedule` independently.

    Parameters
    ----------
    session : SessionV2
    agent_profiles : list of dicts like
        {"agent_id": "a0", "hallucinator": False, "noise": 0.05}
    rng : np.random.Generator

    Returns
    -------
    dict attribute -> list[Commitment], one commitment per agent per
    attribute, in `agent_profiles` order (identical order for every attribute).
    """
    by_attr: dict[str, list[Commitment]] = {a: [] for a in session.schedule}
    for profile in agent_profiles:
        agent_id = str(profile["agent_id"])
        hallucinator = bool(profile.get("hallucinator", False))
        noise = float(profile.get("noise", 0.0))
        prior_guesses: dict[str, AgentGuessV2] = {}
        for attribute in session.schedule:
            g = synthetic_agent_guess(
                session, attribute, prior_guesses, hallucinator, rng
            )
            p = float(np.clip(g.probability + rng.normal(0.0, noise), 0.05, 0.95))
            jittered = AgentGuessV2(attribute, g.value, p)
            prior_guesses[attribute] = jittered  # own cascade, jittered report
            by_attr[attribute].append(Commitment(agent_id, attribute, g.value, p))
    return by_attr
