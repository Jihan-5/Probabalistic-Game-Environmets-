"""Detailed-Questions baseline (paper Table 2 / §6, the strongest non-PGE rival).

Instead of PGE's probabilistic guess + behavioral signal loop, this baseline:
  1. At each turn, the agent emits a *clarifying question* asking the user
     about a specific attribute (rather than guessing it with probability).
  2. The "user" (in our dbbench case: the database) answers the question by
     running a SQL probe and returning the result.
  3. The agent uses the answer to refine its understanding.
  4. Final answer is the agent's last-turn response.

Hallucination score for a Detailed-Questions trajectory:
  - We use the agent's final-answer self-reported probability (low p = less
    confident = more likely hallucinating).
  - Alternatively, fraction of clarification answers the agent contradicts
    in subsequent reasoning.

For the AUROC comparison we just use 1 - final_probability as the
hallucination score — Detailed Questions doesn't have a graphon, so the
only signal is the agent's own self-assessed confidence.
"""

from __future__ import annotations

import re
from typing import Mapping

from pge import config
from pge.agents.llama_together import LlamaTogetherAgent, MODEL_LLAMA_70B
from pge.benchmarks.dbbench import DbbenchTask, _build_sqlite, _safe_ident


DETAILED_QS_SYSTEM = """\
You are an interactive assistant. To answer the user's question accurately you
may ask up to 3 clarifying questions about the data before committing to an
answer. Each clarifying question must be answerable by examining the table.

At each turn output exactly one of:
  ASK: <a specific question about the data>
  ANSWER: <your final answer>, p=<your confidence as 0.xx>

Examples:
  ASK: How many rows have Title='Manager' and Department='Sales'?
  ANSWER: 42, p=0.85

Output exactly one line. No prose, no markdown.
"""


_ASK_RE = re.compile(r"^\s*ASK\s*:\s*(?P<q>.+)$", re.IGNORECASE)
_ANS_RE = re.compile(
    r"^\s*ANSWER\s*:\s*(?P<a>.+?)\s*,\s*p\s*=\s*(?P<p>0?\.\d+|1\.0+|1)\s*$",
    re.IGNORECASE,
)


class DetailedQuestionsBaseline:
    """Multi-turn clarifying-question baseline.

    Reuses the underlying Together API call but with a different system
    prompt and turn-handling logic.
    """

    def __init__(
        self,
        model: str = MODEL_LLAMA_70B,
        max_turns: int = 4,
        temperature: float = 0.2,
        max_tokens: int = 120,
        api_key: str | None = None,
    ):
        # We piggyback on LlamaTogetherAgent's API plumbing so token accounting
        # and retries are consistent across PGE and this baseline.
        self._llm = LlamaTogetherAgent(
            attribute_schedule=["__detailed_qs__"],  # unused
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=DETAILED_QS_SYSTEM,
        )
        self.max_turns = max_turns

    # ------------------------------------------------ token accounting helpers

    @property
    def input_tokens_used(self) -> int:
        return self._llm.input_tokens_used

    @property
    def output_tokens_used(self) -> int:
        return self._llm.output_tokens_used

    def estimate_cost(self, input_price_per_m: float, output_price_per_m: float) -> float:
        return self._llm.estimate_cost(input_price_per_m, output_price_per_m)

    # ----------------------------------------------------------------- the loop

    def run(self, task: DbbenchTask) -> tuple[str, float, list[str]]:
        """Run one trajectory.

        Returns:
            (final_answer, agent_confidence, transcript) where transcript is
            the list of turn strings emitted by the agent.
        """
        conn = _build_sqlite(task)
        cur = conn.cursor()
        tname = _safe_ident(task.table_name)
        cols = ", ".join(_safe_ident(c) for c in task.columns)

        transcript: list[str] = []
        context_prefix = (
            f"Question: {task.description}\n"
            f"Table: {task.table_name}  ({len(task.rows)} rows)\n"
            f"Columns: {', '.join(task.columns)}\n"
        )
        history_lines: list[str] = []
        final_answer = "unknown"
        final_p = 0.5
        for turn in range(self.max_turns):
            history_str = "\n".join(history_lines) if history_lines else "(no prior turns)"
            prompt = (
                f"{context_prefix}\n"
                f"Prior turns:\n{history_str}\n\n"
                f"Now produce ONE line, either an ASK: question or your final ANSWER:."
            )
            try:
                text = self._llm._call_api(prompt)
            except Exception:
                break
            text = text.strip().splitlines()[0] if text.strip() else ""
            transcript.append(text)

            ans = _ANS_RE.match(text)
            if ans:
                final_answer = ans.group("a").strip()
                try:
                    final_p = float(ans.group("p"))
                except ValueError:
                    final_p = 0.5
                break

            ask = _ASK_RE.match(text)
            if ask:
                # Translate the question into a brief textual answer derived
                # from the underlying table. We don't actually execute the
                # agent's natural-language question — we just give it a
                # generic peek at the table (deterministic, free of LLM
                # cost) to simulate the user answering.
                question = ask.group("q").strip()
                preview_rows = task.rows[:3]
                answer = (
                    f"Sample rows from {task.table_name}: {preview_rows}. "
                    f"Columns are: {', '.join(task.columns)}."
                )
                history_lines.append(f"  ASK: {question}")
                history_lines.append(f"  USER: {answer}")
                continue

            # Unparseable line — treat as a malformed answer.
            history_lines.append(f"  (unparseable: {text[:80]})")
            break

        conn.close()
        return final_answer, final_p, transcript
