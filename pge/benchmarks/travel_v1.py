"""Synthetic travel-booking environment v1: the second private-attribute domain.

Same contract as pge.benchmarks.healthcare_v2 (every detector, label and
analysis runs on either domain unchanged):

- A traveler answers 5 coarse INTAKE questions (visible to the agent).
- The traveler has 4 undisclosed HIDDEN facts (loyalty tier, mobility needs,
  dietary requirement, seat preference), independent of the intake and of
  each other, with small discrete supports (36 joint combinations).
- The agent commits to 12 categorical BOOKING attributes. Each attribute's
  true value follows a deterministic agency rule (not shown to the agent) of
  the intake, the hidden facts and earlier attributes ("attr:" deps). Six
  attributes are intake-only (rule-determined); six depend on hidden facts
  directly or transitively, some only conditionally on the intake (e.g.
  lounge access is fixed by a business fare or a low budget, and depends on
  the loyalty tier otherwise).
- Detector-visible outcome channel: `noisy_feedback`, the truth-consistency
  bit flipped with probability eps, and 0 (unobserved) for the two
  NO_FEEDBACK_ATTRS the traveler never reacts to. Labels use noise-free truth.
- Sessions schedule a dependency-closed random subset of 6-12 attributes in
  canonical order; task ids are "tv1_<i>".

The synthetic agents at the bottom (free, no LLM) mirror healthcare_v2's and
are used for sanity checks of the label base rates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory, Turn

DOMAIN = "travel_v1"
TASK_PREFIX = "tv1"
USER_ROLE = "traveler"  # who reacts to the agent's commitments (agent-facing wording)

# ------------------------------------------------------------------ schema

# Intake questions (visible): field -> (values, sampling probabilities).
SETUP_DIST_T1: dict[str, tuple[list[str], list[float]]] = {
    "trip_type": (["business", "leisure", "family"], [0.35, 0.45, 0.20]),
    "budget": (["low", "medium", "high"], [0.30, 0.50, 0.20]),
    "travelers": (["1", "2", "3_plus"], [0.40, 0.35, 0.25]),
    "duration": (["short", "long"], [0.60, 0.40]),
    "booking_window": (["last_minute", "normal", "early"], [0.25, 0.50, 0.25]),
}
SETUP_QUESTIONS_T1 = list(SETUP_DIST_T1)

# Hidden traveler facts (NOT exposed in the intake): field -> (values, prior).
HIDDEN_DIST_T1: dict[str, tuple[list[str], list[float]]] = {
    "loyalty_tier": (["none", "silver", "gold"], [0.60, 0.25, 0.15]),
    "mobility_needs": (["no", "yes"], [0.85, 0.15]),
    "dietary": (["none", "vegetarian", "other"], [0.65, 0.20, 0.15]),
    "seat_preference": (["aisle", "window"], [0.55, 0.45]),
}
HIDDEN_FIELDS = list(HIDDEN_DIST_T1)

# attribute -> (candidate values, dependencies), in canonical order (every
# "attr:" dependency precedes its child). Dependencies name intake keys,
# hidden keys ("hidden:") or earlier attributes ("attr:").
ATTRIBUTES_T1: dict[str, tuple[list[str], list[str]]] = {
    "fare_class": (["economy", "premium", "business"], ["trip_type", "budget"]),
    "lounge_access": (["yes", "no"], ["attr:fare_class", "budget", "hidden:loyalty_tier"]),
    "refundable_fare": (["yes", "no"], ["trip_type", "booking_window"]),
    "wheelchair_assistance": (["yes", "no"], ["hidden:mobility_needs"]),
    "checked_bags": (["none", "one", "two"], ["trip_type", "duration", "travelers"]),
    "seat_assignment": (["aisle", "window", "middle"], ["budget", "travelers", "hidden:seat_preference"]),
    "hotel_tier": (["standard", "upscale", "luxury"], ["budget", "trip_type"]),
    "meal_request": (["standard", "vegetarian", "special"], ["duration", "attr:fare_class", "hidden:dietary"]),
    "priority_boarding": (
        ["yes", "no"],
        ["attr:fare_class", "attr:lounge_access", "attr:wheelchair_assistance", "travelers"],
    ),
    "extra_legroom": (["yes", "no"], ["attr:fare_class", "duration", "hidden:loyalty_tier"]),
    "travel_insurance": (["yes", "no"], ["attr:refundable_fare", "duration", "trip_type"]),
    "airport_transfer": (
        ["none", "shuttle", "private_car"],
        ["attr:hotel_tier", "attr:checked_bags", "travelers"],
    ),
}

# Attributes the traveler never reacts to (no observable feedback).
NO_FEEDBACK_ATTRS = frozenset({"travel_insurance", "priority_boarding"})

# Descriptions shown to LLM agents: allowed values only, no derivation hints.
ATTR_DESCRIPTIONS_T1 = {
    "fare_class": "Cabin class of the flight booking. Output exactly one of: economy, premium, business.",
    "lounge_access": "Should airport lounge access be included in the booking? Output exactly 'yes' or 'no'.",
    "refundable_fare": "Should the booked fare be refundable? Output exactly 'yes' or 'no'.",
    "wheelchair_assistance": "Should wheelchair assistance be requested at the airport? Output exactly 'yes' or 'no'.",
    "checked_bags": "Checked baggage to add to the booking. Output exactly one of: none, one, two.",
    "seat_assignment": "Seat to assign to the traveler. Output exactly one of: aisle, window, middle.",
    "hotel_tier": "Hotel category to book. Output exactly one of: standard, upscale, luxury.",
    "meal_request": "In-flight meal to request. Output exactly one of: standard, vegetarian, special.",
    "priority_boarding": "Should priority boarding be included? Output exactly 'yes' or 'no'.",
    "extra_legroom": "Should an extra-legroom seat be booked? Output exactly 'yes' or 'no'.",
    "travel_insurance": "Should travel insurance be added to the booking? Output exactly 'yes' or 'no'.",
    "airport_transfer": "Ground transfer from the airport to the hotel. Output exactly one of: none, shuttle, private_car.",
}

UNDERDETERMINED = frozenset(
    a for a, (_, deps) in ATTRIBUTES_T1.items() if any(d.startswith("hidden:") for d in deps)
)


# ------------------------------------------------------------- ground truth


def derive_ground_truth(setup: dict, hidden: dict) -> dict[str, str]:
    """The agency's booking rules (EVAL ONLY; never shown to the agent)."""
    trip, budget, trav = setup["trip_type"], setup["budget"], setup["travelers"]
    dur, window = setup["duration"], setup["booking_window"]
    tier, mob = hidden["loyalty_tier"], hidden["mobility_needs"]
    diet, pref = hidden["dietary"], hidden["seat_preference"]

    t: dict[str, str] = {}
    t["fare_class"] = (
        "business"
        if (trip == "business" and budget == "high")
        else "premium"
        if (budget == "high" or (trip == "business" and budget == "medium"))
        else "economy"
    )
    # Included with a business fare; excluded by basic (low-budget) fares;
    # otherwise granted by loyalty status.
    t["lounge_access"] = (
        "yes"
        if t["fare_class"] == "business"
        else "no"
        if budget == "low"
        else "yes"
        if (tier == "gold" or (tier == "silver" and t["fare_class"] == "premium"))
        else "no"
    )
    t["refundable_fare"] = (
        "yes" if (window == "early" or (trip == "business" and window == "normal")) else "no"
    )
    t["wheelchair_assistance"] = "yes" if mob == "yes" else "no"
    t["checked_bags"] = (
        "none"
        if (trip == "business" and dur == "short")
        else "two"
        if (dur == "long" and trav != "1")
        else "one"
    )
    # Basic (low-budget) fares get no seat choice; groups of 3+ are seated
    # together; otherwise the traveler's own preference.
    t["seat_assignment"] = "middle" if (budget == "low" or trav == "3_plus") else pref
    t["hotel_tier"] = (
        "luxury"
        if (budget == "high" and trip != "family")
        else "upscale"
        if (budget == "high" or (budget == "medium" and trip == "business"))
        else "standard"
    )
    # Short trips serve the standard meal only, except in business class.
    t["meal_request"] = (
        "standard"
        if ((dur == "short" and t["fare_class"] != "business") or diet == "none")
        else "vegetarian"
        if diet == "vegetarian"
        else "special"
    )
    t["priority_boarding"] = (
        "yes"
        if (
            t["fare_class"] != "economy"
            or t["lounge_access"] == "yes"
            or t["wheelchair_assistance"] == "yes"
            or trav == "3_plus"
        )
        else "no"
    )
    t["extra_legroom"] = (
        "yes"
        if (t["fare_class"] != "economy" or (dur == "long" and tier != "none"))
        else "no"
    )
    t["travel_insurance"] = (
        "yes" if (t["refundable_fare"] == "no" and (dur == "long" or trip == "family")) else "no"
    )
    t["airport_transfer"] = (
        "private_car"
        if (t["hotel_tier"] == "luxury" or (trav == "3_plus" and t["checked_bags"] == "two"))
        else "shuttle"
        if t["checked_bags"] != "none"
        else "none"
    )
    return t


@dataclass
class SessionT1:
    setup_answers: dict
    hidden_profile: dict
    user_state: dict          # noise-free ground truth (EVAL ONLY)
    schedule: list[str]       # dependency-ordered attributes for this session
    profile_id: int


def sample_session(rng: np.random.Generator, profile_id: int) -> SessionT1:
    setup = {k: str(rng.choice(vals, p=ps)) for k, (vals, ps) in SETUP_DIST_T1.items()}
    hidden = {k: str(rng.choice(vals, p=ps)) for k, (vals, ps) in HIDDEN_DIST_T1.items()}
    truth = derive_ground_truth(setup, hidden)

    # Variable-length schedule: 6-12 attributes, closed under attr:
    # dependencies, in canonical order (as healthcare_v2.sample_session_v2).
    all_attrs = list(ATTRIBUTES_T1)
    n_attrs = int(rng.integers(6, 13))
    chosen = set(rng.choice(len(all_attrs), size=n_attrs, replace=False))
    keep = {all_attrs[i] for i in chosen}
    changed = True
    while changed:
        changed = False
        for a in list(keep):
            for dep in ATTRIBUTES_T1[a][1]:
                if dep.startswith("attr:") and dep[5:] not in keep:
                    keep.add(dep[5:])
                    changed = True
    schedule = [a for a in all_attrs if a in keep]
    return SessionT1(setup, hidden, truth, schedule, profile_id)


# ------------------------------------------------------------- labels


@dataclass
class AgentGuessT1:
    attribute: str
    value: str
    probability: float


def eval_label(
    guesses: dict[str, AgentGuessT1],
    session: SessionT1,
    confidence_threshold: float = 0.6,
    error_fraction: float = 0.5,
) -> int:
    """Composite label L (same rule as healthcare_v2.eval_label_v2): the
    session hallucinates iff >= error_fraction of high-confidence guesses are
    wrong w.r.t. NOISE-FREE truth. Eval only."""
    high = [g for g in guesses.values() if g.probability >= confidence_threshold]
    if not high:
        return 0
    wrong = sum(1 for g in high if g.value != session.user_state[g.attribute])
    return int(wrong / len(high) >= error_fraction)


def _schema_user_dependent() -> frozenset:
    """Attributes with a (transitive) hidden dependency in the published
    schema. Detector-side knowledge; uses no truth and no rules."""
    deps = {a: d for a, (_, d) in ATTRIBUTES_T1.items()}
    memo: dict[str, bool] = {}

    def hid(a: str) -> bool:
        if a not in memo:
            memo[a] = False
            memo[a] = any(x.startswith("hidden:") or (x.startswith("attr:") and hid(x[5:]))
                          for x in deps[a])
        return memo[a]

    return frozenset(a for a in deps if hid(a))


USER_DEPENDENT_SCHEMA = _schema_user_dependent()

# Support of the hidden profile in sample_session (canonical product order).
_HIDDEN_SUPPORT = tuple(
    dict(zip(HIDDEN_FIELDS, combo))
    for combo in itertools.product(*(vals for vals, _ in HIDDEN_DIST_T1.values()))
)


def _prior_weight(hidden: dict) -> float:
    w = 1.0
    for k, (vals, ps) in HIDDEN_DIST_T1.items():
        w *= ps[vals.index(hidden[k])]
    return w


# Simulator prior over the hidden profile: [(hidden_dict, weight)], whole
# support, weights sum to 1. ORACLE information (analysis only).
HIDDEN_PRIOR: list[tuple[dict, float]] = [(h, _prior_weight(h)) for h in _HIDDEN_SUPPORT]


@lru_cache(maxsize=None)
def _user_dependent_cached(setup_items: tuple) -> frozenset:
    setup = dict(setup_items)
    seen: dict[str, set] = {a: set() for a in ATTRIBUTES_T1}
    for hidden in _HIDDEN_SUPPORT:
        for a, v in derive_ground_truth(setup, hidden).items():
            seen[a].add(v)
    return frozenset(a for a, v in seen.items() if len(v) > 1)


def user_dependent_attrs(setup_answers: dict) -> frozenset:
    """Attributes whose TRUE value is not determined by this session's intake
    answers (it varies over the hidden profile's support). EVAL ONLY: uses
    the agency's rules."""
    return _user_dependent_cached(tuple(sorted(setup_answers.items())))


def eval_label_user(
    guesses: dict[str, AgentGuessT1],
    session: SessionT1,
    confidence_threshold: float = 0.6,
    error_fraction: float = 0.5,
) -> int:
    """User-specific label L_U: the eval_label rule restricted to confident
    commitments whose truth depends on undisclosed traveler facts in THIS
    session (user_dependent_attrs). 0 if there are none. Eval only."""
    U = user_dependent_attrs(session.setup_answers)
    high = [g for g in guesses.values() if g.probability >= confidence_threshold and g.attribute in U]
    if not high:
        return 0
    wrong = sum(1 for g in high if g.value != session.user_state[g.attribute])
    return int(wrong >= math.ceil(error_fraction * len(high)))


def dag_from_trajectory(traj: Trajectory) -> ChainOfThoughtDAG:
    """Rebuild the session DAG offline from a cached travel-v1 trajectory
    (dependency edges from ATTRIBUTES_T1 + sequential backbone; node scores
    and GuessNode.signal from the noisy_feedback channel)."""
    G = ChainOfThoughtDAG()
    node_idx: dict[str, int] = {}
    for turn in traj.turns:
        signal = turn.signals.get("noisy_feedback", 0)
        idx = G.add_node(
            GuessNode(
                attribute=turn.attribute,
                value=turn.guess_value,
                probability=turn.probability,
                score=bayesian_score(turn.probability, signal),
                turn=turn.turn,
                signal=signal,
            )
        )
        node_idx[turn.attribute] = idx
        for dep in ATTRIBUTES_T1.get(turn.attribute, ([], []))[1]:
            if dep.startswith("attr:") and dep[5:] in node_idx:
                parent = node_idx[dep[5:]]
                if parent < idx:
                    G.add_edge(parent, idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)
    return G


# ------------------------------------------------------------ synthetic agent


def synthetic_agent_guess(
    session: SessionT1,
    attribute: str,
    prior_guesses: dict[str, AgentGuessT1],
    hallucinator: bool,
    rng: np.random.Generator,
) -> AgentGuessT1:
    """Same behavioural model as healthcare_v2.synthetic_agent_guess."""
    values, deps = ATTRIBUTES_T1[attribute]
    truth = session.user_state[attribute]
    parent_wrong = any(
        dep.startswith("attr:")
        and dep[5:] in prior_guesses
        and prior_guesses[dep[5:]].value != session.user_state[dep[5:]]
        for dep in deps
    )
    if parent_wrong and rng.random() < 0.6:
        wrong = [v for v in values if v != truth]
        return AgentGuessT1(attribute, str(rng.choice(wrong)), float(rng.uniform(0.7, 0.92)))
    if attribute in UNDERDETERMINED:
        if hallucinator:
            return AgentGuessT1(attribute, str(rng.choice(values)), float(rng.uniform(0.75, 0.95)))
        value = str(rng.choice(values)) if rng.random() < 0.5 else values[0]
        return AgentGuessT1(attribute, value, float(rng.uniform(0.35, 0.55)))
    if rng.random() < 0.9:
        return AgentGuessT1(attribute, truth, float(rng.uniform(0.7, 0.95)))
    wrong = [v for v in values if v != truth]
    return AgentGuessT1(attribute, str(rng.choice(wrong)), float(rng.uniform(0.55, 0.85)))


def run_synthetic_session(
    session: SessionT1,
    hallucinator: bool,
    rng: np.random.Generator,
    epsilon: float = 0.25,
    model: str = "synthetic",
    seed: int = 0,
) -> tuple[Trajectory, ChainOfThoughtDAG]:
    """Simulate one session (same channel and DAG as the LLM protocol)."""
    guesses: dict[str, AgentGuessT1] = {}
    turns: list[Turn] = []
    for t, attribute in enumerate(session.schedule):
        g = synthetic_agent_guess(session, attribute, guesses, hallucinator, rng)
        guesses[attribute] = g
        if attribute in NO_FEEDBACK_ATTRS:
            signal = 0
        else:
            signal = 1 if g.value == session.user_state[attribute] else -1
            if rng.random() < epsilon:
                signal = -signal
        turns.append(Turn(attribute=attribute, guess_value=g.value, probability=g.probability, turn=t,
                          signals={"noisy_feedback": signal}))
    traj = Trajectory(
        task_id=f"{TASK_PREFIX}_{session.profile_id}", domain=DOMAIN, model=model, seed=seed, turns=turns,
        eval_label=eval_label(guesses, session), label_source="programmatic",
        meta={"hallucinator_regime": bool(hallucinator), "epsilon": epsilon,
              "eval_label_user": eval_label_user(guesses, session)},
    )
    return traj, dag_from_trajectory(traj)


def generate_dataset(
    n: int,
    seed: int = 42,
    hallucinator_fraction: float = 0.35,
    epsilon: float = 0.25,
) -> tuple[list[Trajectory], list[ChainOfThoughtDAG]]:
    """Sample n sessions with a mixed synthetic agent population (free)."""
    rng = np.random.default_rng(seed)
    trajs, dags = [], []
    for i in range(n):
        session = sample_session(rng, i)
        hallucinator = bool(rng.random() < hallucinator_fraction)
        traj, G = run_synthetic_session(session, hallucinator, rng, epsilon=epsilon, seed=seed)
        trajs.append(traj)
        dags.append(G)
    return trajs, dags


# Generic aliases (same names as healthcare_v2's generic aliases).
ATTRIBUTES = ATTRIBUTES_T1
ATTR_DESCRIPTIONS = ATTR_DESCRIPTIONS_T1
SETUP_QUESTIONS = SETUP_QUESTIONS_T1
AgentGuess = AgentGuessT1
Session = SessionT1
