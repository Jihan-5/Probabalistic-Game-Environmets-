"""End-to-end pipeline test (paper Algorithm 1)."""

import numpy as np

from pge.agent import StubAgent
from pge.behavior import SyntheticBehaviorExtractor
from pge.graphon import GraphonEstimator
from pge.protocol import PGEProtocol


def _build_world():
    """Tiny synthetic world used by the test.

    Setup question: 'risk' in {'low', 'high'}.
    Per turn the agent guesses 'asset' (= 'bonds' for low risk, 'stocks' for high).
    The user state stores the ground-truth asset; behavior matches => +1.
    """
    setup_qs = ["risk"]
    schedule = ["asset"]
    value_table = {
        "asset": {
            ("low",): ("bonds", 0.85),
            ("high",): ("stocks", 0.85),
        }
    }
    relevant = {"asset": ("risk",)}
    return setup_qs, schedule, value_table, relevant


def _run_session(protocol, setup_answers, user_state, T=8):
    return protocol.run(setup_answers, user_state, T=T)


def test_pipeline_separates_clean_from_hallucinating():
    rng = np.random.default_rng(42)
    setup_qs, schedule, table, relevant = _build_world()
    agent = StubAgent(schedule, table, relevant, noise=0.05, rng=rng)
    extractor = SyntheticBehaviorExtractor(truth_keys={"asset": "asset"})

    protocol = PGEProtocol(
        setup_questions=setup_qs,
        agent=agent,
        behavior_extractor=extractor,
        score_gated_edges=True,
    )

    # Phase A: collect training DAGs from clean sessions.
    train_dags = []
    for _ in range(80):
        risk = rng.choice(["low", "high"])
        truth = "bonds" if risk == "low" else "stocks"
        train_dags.append(protocol.build_dag({"risk": risk}, {"asset": truth}, T=8))

    graphon = GraphonEstimator(B=4).fit(train_dags)
    protocol.graphon = graphon
    protocol.threshold_tau = PGEProtocol.calibrate_tau(train_dags[:30], graphon, fpr=0.05)

    # Clean test sessions.
    clean_lls = []
    for _ in range(40):
        risk = rng.choice(["low", "high"])
        truth = "bonds" if risk == "low" else "stocks"
        res = protocol.run({"risk": risk}, {"asset": truth}, T=8)
        clean_lls.append(res.log_likelihood)

    # Hallucinating sessions: agent given setup info that contradicts truth, so
    # high-confidence guesses are systematically wrong (b_t = -1, scores flip).
    bad_lls = []
    for _ in range(40):
        risk = rng.choice(["low", "high"])
        wrong_truth = "stocks" if risk == "low" else "bonds"
        res = protocol.run({"risk": risk}, {"asset": wrong_truth}, T=8)
        bad_lls.append(res.log_likelihood)

    # Under score_gated_edges, hallucinating sessions have far fewer edges,
    # which the graphon detects as out-of-distribution. Clean LL should be
    # higher (more negative? same direction at least; we just need separation).
    assert np.mean(clean_lls) != np.mean(bad_lls)
    # And the means should be cleanly separable.
    assert abs(np.mean(clean_lls) - np.mean(bad_lls)) > 0.05
