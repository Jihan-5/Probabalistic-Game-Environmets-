"""Tests for pge.scoring (paper Algorithm 2 / Eqs. 1-2)."""

import numpy as np
import pytest

from pge.scoring import bayesian_score, cumulative_evidence


def test_score_in_range():
    for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
        for b in (-1, 1):
            s = bayesian_score(p, b)
            assert -1.0 <= s <= 1.0


def test_score_monotonic_with_probability_when_consistent():
    # b = +1: higher p -> more positive score
    scores = [bayesian_score(p, 1) for p in [0.1, 0.3, 0.6, 0.9]]
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))


def test_score_sign_flips_with_behavior():
    s_pos = bayesian_score(0.8, 1)
    s_neg = bayesian_score(0.8, -1)
    assert s_pos > 0
    assert s_neg < 0
    assert pytest.approx(s_pos) == -s_neg


def test_score_clipping_at_extremes():
    # Should not blow up at p in {0, 1}.
    s_lo = bayesian_score(0.0, 1)
    s_hi = bayesian_score(1.0, 1)
    assert np.isfinite(s_lo)
    assert np.isfinite(s_hi)
    # Symmetric clipping to (0.01, 0.99): equal magnitudes, opposite signs.
    assert pytest.approx(abs(s_hi), rel=1e-6) == abs(s_lo)
    assert s_hi > 0 and s_lo < 0


def test_score_zero_at_half_probability():
    assert bayesian_score(0.5, 1) == pytest.approx(0.0)
    assert bayesian_score(0.5, -1) == pytest.approx(0.0)


def test_invalid_behavior_signal_rejected():
    with pytest.raises(ValueError):
        bayesian_score(0.5, 0)
    with pytest.raises(ValueError):
        bayesian_score(0.5, 2)


def test_cumulative_evidence_formula():
    scores = [0.8, -0.5, 0.2]
    E_T, var = cumulative_evidence(scores)
    assert E_T == pytest.approx(0.5)
    # Var = sum(1 - |s|)
    assert var == pytest.approx((1 - 0.8) + (1 - 0.5) + (1 - 0.2))


def test_cumulative_evidence_empty():
    E_T, var = cumulative_evidence([])
    assert E_T == 0.0
    assert var == 0.0
