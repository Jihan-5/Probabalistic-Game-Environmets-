"""Synthetic multi-turn healthcare-scheduling environment.

Replaces the dbbench attempt (which was a structural misfit for PGE).
Here we have what PGE was designed for: each session has a *hidden user
state* the agent must infer from a small set of categorical setup answers,
across multiple turns.

Per session:
  - A user profile is sampled (realistic medical decision context).
  - The user answers 5 categorical setup questions (coarse, paper §3.1).
  - Hidden ground-truth attributes are deterministically derived from the
    profile (some attributes need fine-grained info the setup doesn't expose).
  - Across 6 turns the agent guesses one hidden attribute per turn.
  - Behavior signals: +1 if agent's guess matches the hidden ground truth,
    -1 otherwise.

The mismatch between coarse setup answers and fine-grained hidden state is
where natural hallucinations arise: the LLM agent will sometimes confidently
guess values not derivable from the setup alone, and PGE should detect this
via the graphon log-likelihood of the resulting DAG.

Hallucination label per session:
  Ground-truth: a session is hallucinating iff the agent gets >=50% of
  the hidden attributes wrong with high confidence (p >= 0.6).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from pge.agent import Guess
from pge.behavior import BehaviorExtractor


SETUP_QUESTIONS = [
    "recent_treatment",       # yes / no
    "chronic_conditions",     # none / some / many
    "urgency",                # low / medium / high
    "known_allergies",        # none / contrast / iodine / multiple
    "age_group",              # under_40 / 40_to_65 / over_65
]


# Per-turn attribute schedule (6 turns).
HEALTHCARE_ATTRIBUTES = [
    "scan_safe_now",          # yes / no (depends on recent_treatment + days)
    "needs_contrast",         # yes / no (depends on scan_type)
    "contrast_risk_level",    # low / medium / high (depends on allergies + age)
    "schedule_priority",      # routine / soon / urgent (depends on urgency + age)
    "requires_specialist",    # yes / no (depends on chronic_conditions)
    "drug_interaction_concern",  # yes / no (depends on chronic_conditions + recent_treatment)
]


# Map setup answers -> deterministic ground truth for each hidden attribute.
# This is the "policy" the LLM agent is implicitly trying to learn.
def derive_ground_truth(setup_answers: Mapping[str, str]) -> dict[str, str]:
    rt = setup_answers["recent_treatment"]
    cc = setup_answers["chronic_conditions"]
    urg = setup_answers["urgency"]
    allergy = setup_answers["known_allergies"]
    age = setup_answers["age_group"]

    return {
        # If you had recent treatment, a scan within 7 days isn't safe.
        "scan_safe_now": "no" if rt == "yes" else "yes",

        # Contrast is needed for most diagnostic scans by default.
        "needs_contrast": "yes" if urg in {"medium", "high"} else "no",

        # Risk: anyone with contrast/iodine/multiple allergies is at least medium;
        # over-65 with any allergy is high.
        "contrast_risk_level": (
            "high"   if (allergy in {"contrast", "iodine", "multiple"} and age == "over_65")
            else "medium" if allergy != "none"
            else "low"
        ),

        # Priority depends on urgency, escalated by age.
        "schedule_priority": (
            "urgent" if urg == "high"
            else "soon" if (urg == "medium" or age == "over_65")
            else "routine"
        ),

        # Specialist needed for any chronic condition.
        "requires_specialist": "no" if cc == "none" else "yes",

        # Drug interactions if multiple conditions AND recent treatment.
        "drug_interaction_concern": "yes" if (cc == "many" and rt == "yes") else "no",
    }


@dataclass
class HealthcareSession:
    setup_answers: dict[str, str]
    user_state: dict[str, str]
    profile_id: int


def sample_session(rng: np.random.Generator, profile_id: int) -> HealthcareSession:
    rt = str(rng.choice(["yes", "no"], p=[0.4, 0.6]))
    cc = str(rng.choice(["none", "some", "many"], p=[0.5, 0.35, 0.15]))
    urg = str(rng.choice(["low", "medium", "high"], p=[0.5, 0.3, 0.2]))
    allergy = str(rng.choice(["none", "contrast", "iodine", "multiple"], p=[0.7, 0.1, 0.1, 0.1]))
    age = str(rng.choice(["under_40", "40_to_65", "over_65"], p=[0.4, 0.4, 0.2]))
    setup = {
        "recent_treatment": rt,
        "chronic_conditions": cc,
        "urgency": urg,
        "known_allergies": allergy,
        "age_group": age,
    }
    truth = derive_ground_truth(setup)
    return HealthcareSession(setup_answers=setup, user_state=truth, profile_id=profile_id)


def sample_dataset(n: int, seed: int = 42) -> list[HealthcareSession]:
    rng = np.random.default_rng(seed)
    return [sample_session(rng, i) for i in range(n)]


# --------------------------------------------------- behavior extractor


class HealthcareBehaviorExtractor(BehaviorExtractor):
    """Strict match between agent's guess value and hidden ground truth.

    Returns +1 if values match (after normalization), -1 otherwise.
    """

    def extract(self, guess: Guess, user_state: Mapping[str, object], turn: int) -> int:
        truth = str(user_state.get(guess.attribute, "")).strip().lower()
        claim = str(guess.value).strip().lower()
        if not truth:
            return -1  # unknown attribute -> hallucination
        # accept short forms / substring containment
        return 1 if (claim == truth or claim in truth or truth in claim) else -1


def session_is_hallucinating(
    dag,
    session: HealthcareSession,
    confidence_threshold: float = 0.6,
    error_fraction: float = 0.5,
) -> bool:
    """A session is labeled hallucinating iff the agent made high-confidence
    wrong guesses for at least `error_fraction` of turns.

    This matches paper §3.2's intuition: hallucination = confident incorrect
    assertions, not just incorrect ones.
    """
    n_high_conf = 0
    n_high_conf_wrong = 0
    for node in dag.nodes:
        if node.probability < confidence_threshold:
            continue
        n_high_conf += 1
        truth = str(session.user_state.get(node.attribute, "")).strip().lower()
        claim = str(node.value).strip().lower()
        matches = (claim == truth or claim in truth or truth in claim) and bool(truth)
        if not matches:
            n_high_conf_wrong += 1
    if n_high_conf == 0:
        return False  # agent was uncertain throughout — not hallucinating per spec
    return (n_high_conf_wrong / n_high_conf) >= error_fraction


# Attribute descriptions intentionally OMIT the ground-truth derivation rules
# (they used to include "(Hint: ...)" but that made the task trivial for 70B).
# The LLM must reason from its own medical knowledge applied to the setup
# answers — which is where natural hallucinations arise.
ATTR_DESCRIPTIONS = {
    "scan_safe_now": (
        "Is it safe for this patient to receive a CT/MRI scan today? "
        "Output exactly 'yes' or 'no'."
    ),
    "needs_contrast": (
        "Will the diagnostic scan require contrast agent? "
        "Output exactly 'yes' or 'no'."
    ),
    "contrast_risk_level": (
        "Risk level for adverse reaction to contrast agent for this patient. "
        "Output exactly one of: low, medium, high."
    ),
    "schedule_priority": (
        "Scheduling priority for this patient. "
        "Output exactly one of: routine, soon, urgent."
    ),
    "requires_specialist": (
        "Should the patient be routed to a specialist for follow-up? "
        "Output exactly 'yes' or 'no'."
    ),
    "drug_interaction_concern": (
        "Is there a concern about drug-drug interactions for this patient? "
        "Output exactly 'yes' or 'no'."
    ),
}
