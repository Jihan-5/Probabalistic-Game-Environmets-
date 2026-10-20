"""Supervised session detectors from the agent's own commitments (added after
the 2026-10-20 blind review; frozen in results/PREREG_confirmatory.md).

ContextLR(with_feedback=False)  "ctx_lr"
    What the CONTEXT ALONE reveals: logistic regression on counts of
    confident (p >= 0.6) commitments per (attribute, value) cell, plus |H|,
    the number of turns and mean stated confidence. No feedback, no samples.
    It learns population base rates (which commitments tend to be wrong);
    it cannot see the individual user.
ContextLR(with_feedback=True)   "ctx_lr+fb"
    Same, plus counts per (attribute, value, feedback signal) cell: the
    strongest generic supervised use of the same inputs PGE sees.
PriorOnly(posterior)            "<name>-prior"
    Fits a label-event posterior (V3/V6) normally, then scores with every
    commitment's likelihood ratio set to 1: the same fitted priors and label
    event, with THIS user's reactions ignored. posterior - PriorOnly(posterior)
    is the pre-registered feedback increment. (For V6, setting signals to 0 is
    NOT equivalent: V6 models "no reaction" as evidence.)
FeedbackRemoved(det)            "<det.name>-fb"
    Fits `det` normally, then scores with every signal set to 0. For generic
    detectors only; for V3 it equals PriorOnly, for V6 it does not.

Feature definition matches the blind review's "LR cells" check
(scratchpad review_checks.py): standardized features, L2 logistic
regression with C = 0.1, fixed (no tuning).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from pge.baselines.base import TrajectoryScorer
from pge.baselines.posterior import label_threshold, poisson_binomial_tail, turn_signal

HIGH = 0.6


class ContextLR(TrajectoryScorer):
    def __init__(self, with_feedback: bool = False, C: float = 0.1, high: float = HIGH):
        self.with_feedback, self.C, self.high = with_feedback, C, high
        self.name = "ctx_lr+fb" if with_feedback else "ctx_lr"
        self.vocab: dict = {}
        self.mu = self.sd = self.model = None
        self.const: Optional[float] = None

    def _keys(self, turn):
        yield ("c", turn.attribute, turn.guess_value)
        if self.with_feedback:
            yield ("f", turn.attribute, turn.guess_value, turn_signal(turn))

    def _features(self, traj) -> np.ndarray:
        x = np.zeros(len(self.vocab))
        for turn in traj.turns:
            if turn.probability < self.high:
                continue
            for k in self._keys(turn):
                if k in self.vocab:
                    x[self.vocab[k]] += 1
        h = sum(t.probability >= self.high for t in traj.turns)
        extra = [h, len(traj.turns), float(np.mean([t.probability for t in traj.turns]))]
        return np.concatenate([x, extra])

    def fit(self, trajectories, dags=None, labels=None):
        if labels is None:
            raise ValueError(f"{self.name} is supervised; pass labels")
        y = np.asarray(labels, dtype=int)
        keys = {k for t in trajectories for turn in t.turns if turn.probability >= self.high
                for k in self._keys(turn)}
        self.vocab = {k: i for i, k in enumerate(sorted(keys, key=str))}
        X = np.stack([self._features(t) for t in trajectories])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        if len(set(y.tolist())) < 2:  # degenerate train split: constant score
            self.const = float(y.mean())
            return self
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mu) / self.sd, y)
        return self

    def score(self, trajectory, dag=None) -> float:
        if self.const is not None:
            return self.const
        x = (self._features(trajectory) - self.mu) / self.sd
        return float(self.model.predict_proba(x[None, :])[0, 1])


def without_feedback(traj):
    """Copy of `traj` with every signal set to 0 (unobserved)."""
    t2 = copy.deepcopy(traj)
    for turn in t2.turns:
        turn.signals = {k: 0 for k in turn.signals}
    return t2


class FeedbackRemoved(TrajectoryScorer):
    def __init__(self, det: TrajectoryScorer):
        self.det = det
        self.name = f"{det.name}-fb"

    def fit(self, trajectories, dags=None, labels=None):
        self.det.fit(trajectories, dags, labels)
        return self

    def score(self, trajectory, dag=None) -> float:
        return self.det.score(without_feedback(trajectory))


class PriorOnly(TrajectoryScorer):
    def __init__(self, posterior):
        self.det = posterior
        self.name = f"{posterior.name}-prior"

    def fit(self, trajectories, dags=None, labels=None):
        self.det.fit(trajectories, dags, labels)
        return self

    def score(self, trajectory, dag=None) -> float:
        q = [self.det._prior(t) for t in trajectory.turns if self.det._in_event(t)]
        if not q:
            return 0.0
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), self.det.frac))
