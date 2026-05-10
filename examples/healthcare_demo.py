"""Synthetic healthcare-scheduling demo for the PGE pipeline.

Runs:
  1. Generate ~500 clean multi-turn sessions, fit a sparse graphon.
  2. Calibrate threshold tau at 5% FPR on a held-out clean batch.
  3. Test on 100 clean + 100 "hallucinating" sessions (the agent receives
     wrong setup info, so its high-confidence guesses misalign with user
     behavior).
  4. Report AUROC for log-likelihood-based hallucination detection.

This is a *toy* demo — no real LLM, no real EMR data — meant to confirm the
pipeline is wired correctly. It is not expected to reach the paper's 0.961
AUROC; that requires Llama3-70B + AgentBench, which is out of scope here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from pge.agent import StubAgent
from pge.behavior import SyntheticBehaviorExtractor
from pge.graphon import GraphonEstimator
from pge.protocol import PGEProtocol


# ----------------------------------------------------------------- world setup

SETUP_QUESTIONS = ["recent_treatment", "chronic_conditions", "urgency"]


# Per-turn the agent guesses one of these attributes; each guess depends on
# specific setup answers via `relevant_keys`.
ATTRIBUTE_SCHEDULE = [
    "scan_safe_now",       # depends on recent_treatment
    "needs_followup",      # depends on chronic_conditions
    "schedule_priority",   # depends on urgency
    "scan_safe_now",
    "needs_followup",
    "schedule_priority",
]

VALUE_TABLE = {
    "scan_safe_now": {
        ("yes",): ("no", 0.85),   # recent treatment -> not safe
        ("no",): ("yes", 0.85),
    },
    "needs_followup": {
        ("none",): ("no", 0.80),
        ("some",): ("yes", 0.70),
        ("many",): ("yes", 0.90),
    },
    "schedule_priority": {
        ("low",): ("routine", 0.80),
        ("medium",): ("soon", 0.70),
        ("high",): ("urgent", 0.90),
    },
}

RELEVANT = {
    "scan_safe_now": ("recent_treatment",),
    "needs_followup": ("chronic_conditions",),
    "schedule_priority": ("urgency",),
}

TRUTH_KEYS = {
    "scan_safe_now": "scan_safe_now",
    "needs_followup": "needs_followup",
    "schedule_priority": "schedule_priority",
}


# ------------------------------------------------------------------ generators


@dataclass
class Session:
    setup_answers: dict
    user_state: dict


def sample_clean_session(rng: np.random.Generator) -> Session:
    rt = rng.choice(["yes", "no"])
    cc = rng.choice(["none", "some", "many"])
    urg = rng.choice(["low", "medium", "high"])
    setup = {"recent_treatment": rt, "chronic_conditions": cc, "urgency": urg}
    truth = {
        "scan_safe_now": "no" if rt == "yes" else "yes",
        "needs_followup": "no" if cc == "none" else "yes",
        "schedule_priority": {"low": "routine", "medium": "soon", "high": "urgent"}[urg],
    }
    return Session(setup_answers=setup, user_state=truth)


def sample_hallucinating_session(rng: np.random.Generator) -> Session:
    """Setup answers describe one user; ground truth is a different one.

    Models the paper's failure mode: the agent confidently propagates
    user-attribute guesses derived from missing/incorrect setup context.
    """
    visible = sample_clean_session(rng)
    hidden = sample_clean_session(rng)
    return Session(setup_answers=visible.setup_answers, user_state=hidden.user_state)


# ------------------------------------------------------------------ demo glue


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC via Mann-Whitney U.

    Higher score => more likely positive class. With the convention that
    `scores_pos` are scores for known-positive instances, AUROC = P(score_pos
    > score_neg) over random pairs.
    """
    from scipy.stats import rankdata

    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)  # ascending: higher s => higher rank
    U = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    agent = StubAgent(
        attribute_schedule=ATTRIBUTE_SCHEDULE,
        value_table=VALUE_TABLE,
        relevant_keys=RELEVANT,
        noise=args.noise,
        rng=rng,
    )
    extractor = SyntheticBehaviorExtractor(truth_keys=TRUTH_KEYS)

    # We use a single PGEProtocol instance for everything; graphon/tau are
    # filled in after the training phase. score_gated_edges=True makes
    # successive reasoning edges depend on the previous turn being confirmed
    # by user behavior — under hallucination this collapses the chain and
    # leaves a structural fingerprint for the graphon to detect.
    protocol = PGEProtocol(
        setup_questions=SETUP_QUESTIONS,
        agent=agent,
        behavior_extractor=extractor,
        score_gated_edges=True,
    )

    # ---- Phase 1: training data (clean sessions) -> graphon
    print(f"Generating {args.n_train} clean training sessions...")
    train_dags = [
        protocol.build_dag(s.setup_answers, s.user_state, args.turns)
        for s in (sample_clean_session(rng) for _ in range(args.n_train))
    ]
    graphon = GraphonEstimator(B=args.blocks).fit(train_dags)
    protocol.graphon = graphon

    # ---- Phase 2: calibrate tau at 5% FPR on a held-out clean batch
    val_dags = [
        protocol.build_dag(s.setup_answers, s.user_state, args.turns)
        for s in (sample_clean_session(rng) for _ in range(200))
    ]
    tau = PGEProtocol.calibrate_tau(val_dags, graphon, fpr=0.05)
    protocol.threshold_tau = tau
    print(f"Calibrated tau at 5% FPR: {tau:.3f}")

    # ---- Phase 3: test
    print(f"Running {args.n_test} clean + {args.n_test} hallucinating test sessions...")
    clean_lls, bad_lls = [], []
    clean_flag, bad_flag = 0, 0
    for _ in range(args.n_test):
        s = sample_clean_session(rng)
        res = protocol.run(s.setup_answers, s.user_state, T=args.turns)
        clean_lls.append(res.log_likelihood)
        clean_flag += int(res.hallucinated)
    for _ in range(args.n_test):
        s = sample_hallucinating_session(rng)
        res = protocol.run(s.setup_answers, s.user_state, T=args.turns)
        bad_lls.append(res.log_likelihood)
        bad_flag += int(res.hallucinated)

    clean_lls = np.array(clean_lls)
    bad_lls = np.array(bad_lls)

    auc = auroc(-bad_lls, -clean_lls)  # high score = low log-lik = positive class

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Clean log-lik     : mean={clean_lls.mean():.3f} std={clean_lls.std():.3f}")
    print(f"Hallucinating LL  : mean={bad_lls.mean():.3f} std={bad_lls.std():.3f}")
    print(f"Flag rate (clean) : {clean_flag}/{args.n_test} = {clean_flag / args.n_test:.1%}")
    print(f"Flag rate (bad)   : {bad_flag}/{args.n_test} = {bad_flag / args.n_test:.1%}")
    print(f"AUROC             : {auc:.3f}")


if __name__ == "__main__":
    main()
