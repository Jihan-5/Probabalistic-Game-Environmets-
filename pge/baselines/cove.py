"""Chain-of-Verification baseline (Dhuliawala et al. 2023, adapted).

Failure mode targeted: a hallucinated commitment often fails independent
re-derivation — asked the same question fresh (without its own prior answer
in context), the model lands elsewhere. For each turn we issue one
verification query re-asking the turn's attribute against its original
prompt context (falling back to the trajectory's accumulated reasoning when
the turn has no prompt), parse the re-answer, and score the trajectory as
the fraction of turns where the re-answer disagrees with the committed
value.

The LLM is an injectable `llm_fn(prompt) -> str` — required at
construction, so this module itself makes no API calls. `calls_made`
counts verification queries issued.
"""

from __future__ import annotations

from typing import Callable, Optional

from pge.baselines.base import TrajectoryScorer
from pge.baselines.semantic_entropy import _extract_final_value, _normalize
from pge.trajectory import Trajectory

VERIFY_PROMPT = """Answer the question below from the context alone. \
Do not assume any prior answer is correct.

Context:
{context}

Question: what is the value of "{attribute}"?

Reply with exactly one line:
[final_answer]: <value>, p=<confidence>"""

NO_CONTEXT = "(no additional context)"


class ChainOfVerification(TrajectoryScorer):
    name = "cove"

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        if llm_fn is None:
            raise ValueError("cove requires an llm_fn(prompt) -> str")
        self.llm_fn = llm_fn
        self.calls_made = 0

    # ------------------------------------------------------------- internals

    @staticmethod
    def _fallback_context(trajectory: Trajectory) -> str:
        parts = [t.reasoning.strip() for t in trajectory.turns if t.reasoning.strip()]
        return "\n".join(parts) if parts else NO_CONTEXT

    @staticmethod
    def _parse_reanswer(reply: str) -> str:
        v = _extract_final_value(reply)
        if v:
            return v
        lines = [l.strip() for l in reply.splitlines() if l.strip()]
        return lines[-1] if lines else ""

    # ------------------------------------------------------------- interface

    def score(self, trajectory: Trajectory, dag=None) -> float:
        if not trajectory.turns:
            return 0.0
        disagree = 0
        for t in trajectory.turns:
            context = t.prompt.strip() or self._fallback_context(trajectory)
            prompt = VERIFY_PROMPT.format(context=context, attribute=t.attribute)
            reply = self.llm_fn(prompt)
            self.calls_made += 1
            if _normalize(self._parse_reanswer(reply)) != _normalize(t.guess_value):
                disagree += 1
        return float(disagree / len(trajectory.turns))
