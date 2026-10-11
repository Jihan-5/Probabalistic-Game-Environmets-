"""SelfCheckGPT baseline (Manakul et al. 2023, prompt-variant style).

Failure mode targeted: a hallucinated commitment is poorly supported by the
model's own resamples — the stochastic samples disagree with the committed
answer more often than for grounded claims. Per turn with resamples,
score = fraction of samples whose extracted value does NOT match the
committed `guess_value` (normalized: case/whitespace/punctuation, numeric
equality). Trajectory score = mean over turns with samples; 0 if none.

An injectable `check_fn(claim, sample_text) -> bool` (True = sample
supports the claim) replaces value matching with an NLI/LLM-style check —
no API dependency in this module.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from pge.baselines.base import TrajectoryScorer
from pge.baselines.semantic_entropy import _extract_final_value, _normalize
from pge.trajectory import Trajectory


class SelfCheckGPT(TrajectoryScorer):
    name = "selfcheckgpt"

    def __init__(
        self, check_fn: Optional[Callable[[str, str], bool]] = None
    ):
        self.check_fn = check_fn

    def score(self, trajectory: Trajectory, dag=None) -> float:
        per_turn: list[float] = []
        for t in trajectory.turns:
            if not t.samples:
                continue
            if self.check_fn is not None:
                unsupported = sum(
                    1 for s in t.samples if not self.check_fn(t.guess_value, s)
                )
            else:
                claim = _normalize(t.guess_value)
                # A sample with no extractable value counts as non-support.
                unsupported = sum(
                    1
                    for s in t.samples
                    if _normalize(_extract_final_value(s)) != claim
                )
            per_turn.append(unsupported / len(t.samples))
        if not per_turn:
            return 0.0
        return float(np.mean(per_turn))
