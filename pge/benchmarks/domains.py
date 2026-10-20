"""Registry of private-attribute domains with the shared interface.

Every module in DOMAINS exposes (same semantics across domains):
    DOMAIN, TASK_PREFIX, USER_ROLE, ATTRIBUTES, ATTR_DESCRIPTIONS,
    NO_FEEDBACK_ATTRS, UNDERDETERMINED, USER_DEPENDENT_SCHEMA, HIDDEN_PRIOR,
    derive_ground_truth(setup, hidden), sample_session(rng, profile_id),
    AgentGuess, eval_label, user_dependent_attrs(setup), eval_label_user,
    dag_from_trajectory.

`get` also resolves trajectory domains that reuse another domain's sessions,
truth and labels (the LLM-simulated-patient corpora use healthcare_v2's
session stream, task ids and labels with their own feedback channel).
"""

from __future__ import annotations

from types import ModuleType

from pge.benchmarks import healthcare_v2, travel_v1

DOMAINS: dict[str, ModuleType] = {"healthcare_v2": healthcare_v2, "travel_v1": travel_v1}

# trajectory domain -> domain whose sessions / truth / labels it uses
ALIASES = {"healthcare_llmpatient": "healthcare_v2"}


def get(name: str) -> ModuleType:
    """Domain module for a domain (or trajectory-domain alias) name."""
    key = ALIASES.get(name, name)
    if key not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; known: {sorted(DOMAINS)} (+ aliases {sorted(ALIASES)})")
    return DOMAINS[key]
