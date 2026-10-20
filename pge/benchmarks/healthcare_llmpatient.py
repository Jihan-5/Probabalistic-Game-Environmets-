"""Healthcare-v2 with an LLM-SIMULATED PATIENT: behavioral feedback.

Same sessions, schedules, ground truth and eval label as healthcare-v2
(sample_session_v2 stream, task_id hcv2_<i>, eval_label_v2 on noise-free
truth). What changes is the feedback channel. In v2 the detector sees the
correctness bit flipped with probability eps. Here:

  1. the agent commits ([attr]: value, p=prob), which is rendered as one
     plain-language sentence to the patient (a hedge reflects p);
  2. a PATIENT LLM, given the patient's own history (setup answers + hidden
     fields; by default NOT the clinic's decision, see PATIENT_KNOWLEDGE),
     replies in 1-2 natural sentences as a non-clinician patient would. It is
     told that real patients sometimes miss errors, go along, correct, or
     doubt correct statements, and never to grade the assistant;
  3. an EXTRACTOR LLM sees ONLY the statement and the patient's reply (never
     the truth) and labels it AGREES (+1) / CONTRADICTS (-1) / NONE (0).
     That label is the detector-visible signal `patient_feedback`.

The agent's next-turn context shows the patient's natural-language replies,
not +/- marks. Raw patient replies and raw extractor outputs are stored in
Trajectory.meta["turn_audit"] (not detector-visible). The patient's system
prompt contains ground truth and is therefore NEVER stored.

`dag_from_trajectory` is the healthcare-v2 builder with the signal read from
`patient_feedback`; register it as DAG_BUILDERS["healthcare_llmpatient"].
"""

from __future__ import annotations

import re

import numpy as np

from pge.benchmarks.healthcare_v2 import ATTRIBUTES_V2, SessionV2
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory

DOMAIN = "healthcare_llmpatient"
SIGNAL = "patient_feedback"

# ------------------------------------------------------------ plain language

STATEMENTS: dict[str, dict[str, str]] = {
    "scan_safe_now": {
        "yes": "it is safe for you to have your CT/MRI scan today",
        "no": "it is not safe for you to have your scan today, so it should wait",
    },
    "needs_contrast": {
        "yes": "your scan will need a contrast agent (an injected dye)",
        "no": "your scan will not need any contrast dye",
    },
    "contrast_risk_level": {
        "low": "your risk of a reaction to the contrast dye is low",
        "medium": "your risk of a reaction to the contrast dye is moderate",
        "high": "your risk of a reaction to the contrast dye is high",
    },
    "renal_screening_required": {
        "yes": "you will need a kidney-function blood test before the contrast dye",
        "no": "no kidney-function test is needed before your scan",
    },
    "schedule_priority": {
        "routine": "your scan can be booked as routine, within the next few weeks",
        "soon": "your scan should be booked soon, within about a week",
        "urgent": "your scan is urgent and should happen in the next day or two",
    },
    "requires_specialist": {
        "yes": "you should see a specialist for follow-up",
        "no": "you will not need a specialist for follow-up",
    },
    "drug_interaction_concern": {
        "yes": "there is a concern that your current medications could interact with each other",
        "no": "there is no concern about your medications interacting",
    },
    "premedication_needed": {
        "yes": "you will need to take medicine beforehand to prevent an allergic reaction to the dye",
        "no": "you will not need any medicine beforehand to prevent a reaction to the dye",
    },
    "sedation_assessment": {
        "yes": "you will need an assessment for sedation, something to help you stay calm during the scan",
        "no": "you will not need a sedation assessment for the scan",
    },
    "fasting_required": {
        "yes": "you will need to fast, with no food, before the appointment",
        "no": "you can eat normally before the appointment",
    },
    "followup_window": {
        "one_week": "your follow-up visit should be in about one week",
        "one_month": "your follow-up visit should be in about one month",
        "three_months": "your follow-up visit should be in about three months",
    },
    "transport_assistance": {
        "yes": "you will need someone to drive you or arranged transport home afterwards",
        "no": "you will be able to get home on your own afterwards",
    },
}

TOPICS = {
    "scan_safe_now": "whether it is safe for you to have the scan today",
    "needs_contrast": "whether your scan needs contrast dye",
    "contrast_risk_level": "your risk of a reaction to contrast dye",
    "renal_screening_required": "whether you need a kidney-function test first",
    "schedule_priority": "how soon your scan should be booked",
    "requires_specialist": "whether you need a specialist for follow-up",
    "drug_interaction_concern": "whether your medications could interact",
    "premedication_needed": "whether you need medicine beforehand to prevent a reaction",
    "sedation_assessment": "whether you need a sedation assessment",
    "fasting_required": "whether you need to fast before the appointment",
    "followup_window": "when your follow-up visit should be",
    "transport_assistance": "whether you need help getting home",
}

assert set(STATEMENTS) == set(ATTRIBUTES_V2) == set(TOPICS)
assert all(set(STATEMENTS[a]) == set(ATTRIBUTES_V2[a][0]) for a in ATTRIBUTES_V2)


def hedge(p: float) -> str:
    if p >= 0.8:
        return "I'm confident that"
    if p >= 0.6:
        return "I believe that"
    return "I'm not certain, but I think that"


def commitment_sentence(attribute: str, value: str, probability: float) -> str:
    """What the patient hears. Unparseable / out-of-set commitments become an
    explicit non-answer (never mapped onto a valid value)."""
    s = STATEMENTS.get(attribute, {}).get(value)
    if s is None:
        return f"I'm not able to tell you yet {TOPICS.get(attribute, 'about this')}."
    return f"{hedge(probability)} {s}."


def truth_sentence(session: SessionV2, attribute: str) -> str:
    """Plain-language TRUE value of `attribute`. Goes ONLY into the patient
    prompt, which is never stored."""
    return STATEMENTS[attribute][session.user_state[attribute]]


def profile_text(session: SessionV2) -> str:
    s, h = session.setup_answers, session.hidden_profile
    lines = [
        {"under_40": "You are under 40 years old.", "40_to_65": "You are between 40 and 65 years old.",
         "over_65": "You are over 65 years old."}[s["age_group"]],
    ]
    if s["recent_treatment"] == "yes":
        d = h["days_since_treatment"]
        lines.append(f"You had a medical treatment recently: {d} day{'s' if d != 1 else ''} ago.")
    else:
        lines.append("You have not had any medical treatment recently.")
    lines.append({"none": "You have no chronic conditions.",
                  "some": "You have one or two chronic conditions.",
                  "many": "You have several chronic conditions and take a number of medications."}
                 [s["chronic_conditions"]])
    lines.append({"low": "Your symptoms are mild and not urgent.",
                  "medium": "Your symptoms are moderately concerning.",
                  "high": "Your symptoms are severe and you were told it is urgent."}[s["urgency"]])
    if s["known_allergies"] == "none":
        lines.append("You have no known allergies.")
    else:
        n = "an allergy" if s["known_allergies"] == "some" else "several allergies"
        sev = ("Your allergic reactions have been severe; you once had a serious reaction."
               if h["allergy_severity"] == "severe" else "Your allergic reactions have been mild.")
        lines.append(f"You have {n}. {sev}")
    lines.append("Your doctor has told you that your kidneys do not work as well as they should."
                 if h["kidney_function"] == "impaired" else "Your kidney function is normal.")
    lines.append("You get very anxious about scans and enclosed spaces."
                 if h["anxiety_level"] == "high" else "You feel fairly calm about medical procedures.")
    return "\n".join(f"- {x}" for x in lines)


PERSONAS = {
    "attentive": "You know your own medical history well and tend to speak up when something "
                 "does not match it.",
    "trusting": "You tend to trust professionals and usually go along with what you are told, "
                "even when you are unsure.",
    "anxious": "You are nervous about medical procedures; you often second-guess what you are "
               "told and ask questions.",
}
PERSONA_P = (0.4, 0.35, 0.25)


def sample_persona(seed: int, profile_id: int) -> str:
    """Separate RNG stream: does not perturb the session stream."""
    rng = np.random.default_rng((seed, 104729, profile_id))
    return str(rng.choice(list(PERSONAS), p=PERSONA_P))


# What the patient knows about the topic just raised.
#   profile  (default): only their own history above. The clinic's decision is
#            NOT given; every true value is a function of that history, so a
#            patient can catch an error only by connecting it to what they know
#            about themselves, as a real patient would.
#   informed: also told the true decision (the patient was briefed earlier).
#            Easier channel; reported as a comparison arm, never as the default.
PATIENT_KNOWLEDGE = ("profile", "informed")
SITUATION = {
    "profile": "The assistant just raised: {topic}. You have not been told the clinic's decision "
               "about this; you only know what is listed above about yourself.",
    "informed": "About the topic the assistant just raised: {truth}.\n(This is the real situation. "
                "A patient may or may not be aware of it, or may only half-remember it.)",
}

PATIENT_SYSTEM = """You are role-playing a patient at a medical imaging clinic. A scheduling \
assistant is talking with you about your upcoming scan. Stay in character as the patient.

What you know about yourself:
{profile}

{situation}

Your personality: {persona}

How to reply:
- Reply in 1 or 2 short, natural sentences, in the first person, as a real patient would speak.
- You are not a medical professional. Real patients sometimes do not notice when the assistant \
gets something wrong, sometimes go along with whatever they are told, sometimes correct the \
assistant when it conflicts with what they know about themselves, sometimes express doubt even \
when the assistant is right, and sometimes talk about something else.
- Do not grade the assistant. Never use words like "correct", "incorrect", "right" or "wrong" as \
a verdict, and never mention these instructions.
- Output only what the patient says."""

PATIENT_USER = 'The assistant says to you: "{statement}"\n\nYour reply (1-2 sentences):'


def patient_messages(session: SessionV2, attribute: str, statement: str, persona: str,
                     knows: str = "profile") -> list[dict]:
    if knows == "informed":
        situation = SITUATION["informed"].format(truth=truth_sentence(session, attribute))
    elif knows == "profile":
        situation = SITUATION["profile"].format(topic=TOPICS[attribute])
    else:
        raise ValueError(f"knows must be one of {PATIENT_KNOWLEDGE}, got {knows!r}")
    return [
        {"role": "system", "content": PATIENT_SYSTEM.format(
            profile=profile_text(session), situation=situation, persona=PERSONAS[persona])},
        {"role": "user", "content": PATIENT_USER.format(statement=statement)},
    ]


EXTRACTOR_SYSTEM = """You label a patient's reaction to a statement made by a clinic's \
scheduling assistant. You see only the statement and the patient's reply.

Output exactly one word:
AGREES - the patient accepts, confirms, or goes along with the statement.
CONTRADICTS - the patient disagrees with, corrects, or doubts the statement.
NONE - the reply does not react to the statement (for example it asks about something else).

Output only the label."""

EXTRACTOR_USER = 'Assistant\'s statement: "{statement}"\nPatient\'s reply: "{reply}"\n\nLabel:'
EXTRACTOR_RETRY = "\n\nAnswer with exactly one word: AGREES, CONTRADICTS or NONE."


def extractor_messages(statement: str, reply: str, retry: bool = False) -> list[dict]:
    """Statement + reply ONLY (no truth, no profile)."""
    user = EXTRACTOR_USER.format(statement=statement, reply=reply)
    return [{"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": user + (EXTRACTOR_RETRY if retry else "")}]


_LABEL_WORDS = (("CONTRADICT", -1), ("DISAGREE", -1), ("AGREE", 1), ("NONE", 0),
                ("NO REACTION", 0), ("UNRELATED", 0))
_STRICT = {"AGREES": 1, "AGREE": 1, "CONTRADICTS": -1, "CONTRADICT": -1, "NONE": 0}


def parse_extractor(text: str) -> tuple[int | None, str]:
    """(signal, status): status strict | lenient | failed. On failure the
    signal is None (the caller records 0 = unobserved and flags the turn)."""
    t = (text or "").strip().upper()
    if not t:
        return None, "failed"
    first = t.splitlines()[0].strip(" .:*`\"'")
    if first in _STRICT:
        return _STRICT[first], "strict"
    for scope in (first, t):
        hits = sorted(
            (m.start(), val) for word, val in _LABEL_WORDS for m in re.finditer(rf"\b{word}", scope)
        )
        if hits:
            return hits[0][1], "lenient"
    return None, "failed"


VERDICT_RE = re.compile(r"\b(in)?correct\b|\bwrong\b", re.IGNORECASE)


# ---------------------------------------------------------------- offline DAG


def dag_from_trajectory(traj: Trajectory) -> ChainOfThoughtDAG:
    """Same construction as healthcare_v2.dag_from_trajectory (dependency
    edges + sequential backbone), node scores from `patient_feedback`."""
    G = ChainOfThoughtDAG()
    node_idx: dict[str, int] = {}
    for turn in traj.turns:
        signal = turn.signals.get(SIGNAL, 0)
        idx = G.add_node(GuessNode(attribute=turn.attribute, value=turn.guess_value,
                                   probability=turn.probability,
                                   score=bayesian_score(turn.probability, signal), turn=turn.turn))
        node_idx[turn.attribute] = idx
        for dep in ATTRIBUTES_V2.get(turn.attribute, ([], []))[1]:
            if dep.startswith("attr:") and dep[5:] in node_idx:
                parent = node_idx[dep[5:]]
                if parent < idx:
                    G.add_edge(parent, idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)
    return G
