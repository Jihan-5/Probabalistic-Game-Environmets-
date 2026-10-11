"""Tests for the corrected asymmetric-scoring theory (conservative-report lemma)."""

import numpy as np
import pytest

from pge.scoring import (
    ALPHA,
    BETA,
    GAMMA,
    asymmetric_score,
    expected_score,
    implied_belief,
    optimal_report,
    revisitation_threshold,
)


def test_asymmetric_penalty_superlinear():
    # Confident wrong costs beta*p^gamma, more than linear reward alpha*p.
    assert asymmetric_score(0.9, -1) == pytest.approx(-3.0 * 0.81)
    assert asymmetric_score(0.9, 1) == pytest.approx(0.9)
    assert asymmetric_score(0.9, 0) == 0.0


def test_reviewer_qg2c_counterexample():
    # The exact calculation from the ICML'26 review: q=0.5, alpha=1, beta=3,
    # gamma=2 gives p* = q/(6(1-q)) = 1/6 ~= 0.167 — NOT 0.5.
    p_star = optimal_report(0.5)
    assert p_star == pytest.approx(0.5 / (6 * 0.5), rel=1e-9)
    assert p_star == pytest.approx(1.0 / 6.0, rel=1e-9)


def test_optimal_report_strictly_monotone_until_saturation():
    # Strictly increasing on the interior branch; saturates at p*=1 for
    # q >= q_sat = beta*gamma/(alpha+beta*gamma) = 6/7 with defaults.
    q_sat = BETA * GAMMA / (ALPHA + BETA * GAMMA)
    qs = np.linspace(0.01, q_sat - 0.01, 80)
    ps = np.array([optimal_report(q) for q in qs])
    assert np.all(np.diff(ps) > 0)
    assert optimal_report(q_sat + 0.01) == 1.0
    # Non-decreasing globally.
    all_ps = [optimal_report(q) for q in np.linspace(0.01, 0.99, 99)]
    assert np.all(np.diff(all_ps) >= 0)


def test_conservative_region_is_exactly_below_one_minus_alpha_over_betagamma():
    # Corrected lemma: p*(q) < q iff q < 1 - alpha/(beta*gamma) (gamma = 2).
    q_c = 1.0 - ALPHA / (BETA * GAMMA)  # 5/6 with defaults
    for q in np.linspace(0.05, q_c - 0.01, 20):
        assert optimal_report(q) < q
    # Just above the conservative region the optimal report exceeds q.
    assert optimal_report(q_c + 0.01) > q_c + 0.01


def test_implied_belief_inverts_optimal_report():
    for q in np.linspace(0.02, 0.9, 45):
        p = optimal_report(q)
        if p < 1.0:  # inverse only defined on the interior branch
            assert implied_belief(p) == pytest.approx(q, abs=1e-9)


def test_optimal_report_maximizes_expected_score():
    rng = np.random.default_rng(0)
    for q in (0.2, 0.5, 0.8):
        p_star = optimal_report(q)
        best = expected_score(q, p_star)
        for p in rng.uniform(0.0, 1.0, 200):
            assert expected_score(q, float(p)) <= best + 1e-12


def test_boundaries():
    assert optimal_report(0.0) == 0.0
    assert optimal_report(1.0) == 1.0
    assert implied_belief(0.0) == 0.0


def test_gamma_must_exceed_one():
    with pytest.raises(ValueError):
        optimal_report(0.5, gamma=1.0)


def test_revisitation_threshold_scales_with_t():
    tau5 = revisitation_threshold(0.6, 5)
    tau10 = revisitation_threshold(0.6, 10)
    assert tau10 == pytest.approx(2 * tau5)
    assert revisitation_threshold(0.6, 0) == 0.0
    # An honest q_min agent has nonnegative optimal expected score.
    assert tau5 >= 0.0


def test_defaults_match_paper():
    assert (ALPHA, BETA, GAMMA) == (1.0, 3.0, 2.0)
