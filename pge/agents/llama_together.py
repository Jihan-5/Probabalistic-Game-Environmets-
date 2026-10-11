"""Llama-3.1-Instruct agent backed by Together AI.

Implements the `pge.agent.Agent` interface against Together's
OpenAI-compatible chat-completions endpoint. Returns calibrated
probabilistic guesses parseable by the rest of the PGE pipeline.

API reference: https://docs.together.ai/reference/completions-1
"""

from __future__ import annotations

import json
import re
import time
from typing import Mapping, Optional

import requests

from pge import config
from pge.agent import Agent, Guess
from pge.dag import ChainOfThoughtDAG


TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 70B is served serverlessly on Together; as of 2026-09 Together retired all
# serverless Llama-8B endpoints (400 "non-serverless model"), so the 8B tier
# is routed through OpenRouter instead (same Llama family).
MODEL_LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Together, $0.88/M
MODEL_LLAMA_8B = "meta-llama/llama-3.1-8b-instruct"          # OpenRouter, ~$0.03/M

PRICE_PER_M = {
    MODEL_LLAMA_70B: (0.88, 0.88),
    MODEL_LLAMA_8B: (0.05, 0.05),  # conservative upper bound
}

_OPENROUTER_MODELS = {MODEL_LLAMA_8B}


_GUESS_RE = re.compile(
    r"\[(?P<attr>[^\]]+)\]\s*:\s*(?P<val>[^,\n]+?)\s*,\s*p\s*=\s*(?P<prob>0?\.\d+|1\.0+|1)",
    re.IGNORECASE,
)


def _parse_guess(text: str, allowed_attributes: list[str] | None = None) -> Optional[Guess]:
    """Parse "[Attribute]: value, p=0.xx" from arbitrary LLM output.

    Returns None on parse failure so the caller can retry or fall back.
    """
    for match in _GUESS_RE.finditer(text):
        attr = match.group("attr").strip()
        val = match.group("val").strip().strip('"').strip("'")
        try:
            prob = float(match.group("prob"))
        except ValueError:
            continue
        if not 0.0 <= prob <= 1.0:
            continue
        if allowed_attributes is not None and attr not in allowed_attributes:
            continue
        return Guess(attribute=attr, value=val, probability=prob)
    return None


SYSTEM_PROMPT = """\
You are an interactive assistant helping a user with a task. At each turn you
must output exactly one probabilistic guess about a hidden user attribute, in
the strict format:

[<Attribute>]: <value>, p=<probability>

- <Attribute> must be drawn from the provided list of attributes.
- <value> is a short concrete value (one or two words).
- <probability> is your calibrated confidence as a number in (0, 1).

Do not output anything else on the line — no explanation, no extra punctuation.
Output exactly one line.
"""


def _format_history(history: ChainOfThoughtDAG, max_turns: int = 6) -> str:
    if len(history) == 0:
        return "(no prior guesses)"
    lines = []
    for node in history.nodes[-max_turns:]:
        sign = "+" if node.score >= 0 else "-"
        lines.append(
            f"  turn {node.turn}: [{node.attribute}]: {node.value}, "
            f"p={node.probability:.2f}  (feedback: {sign})"
        )
    return "\n".join(lines)


class LlamaTogetherAgent(Agent):
    """LLM-backed agent calling Together's chat-completions API.

    Parameters
    ----------
    attribute_schedule : list[str]
        Order in which to guess attributes across turns. The agent is told to
        guess attribute `attribute_schedule[t % len(...)]` at turn t.
    model : str
        Together model slug. Default Llama-3.3-70B-Instruct-Turbo.
    api_key : str, optional
        Defaults to TOGETHER_API_KEY from .env / env.
    max_retries : int
        Retries on HTTP/parse errors. Each retry burns a few hundred tokens.
    """

    def __init__(
        self,
        attribute_schedule: list[str],
        model: str = MODEL_LLAMA_70B,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        request_timeout: float = 15.0,
        max_tokens: int = 64,
        temperature: float = 0.3,
        attribute_descriptions: Optional[dict[str, str]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.attribute_schedule = attribute_schedule
        self.model = model
        if model in _OPENROUTER_MODELS:
            self.base_url = OPENROUTER_URL
            self.api_key = api_key or config.require("OPENROUTER_API_KEY")
        else:
            self.base_url = TOGETHER_URL
            self.api_key = api_key or config.require("TOGETHER_API_KEY")
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.attribute_descriptions = attribute_descriptions or {}
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        # Token bookkeeping for budget enforcement.
        self.input_tokens_used = 0
        self.output_tokens_used = 0
        self.calls_made = 0

    # -------------------------------------------------------- Agent interface

    def guess(
        self,
        turn: int,
        setup_answers: Mapping[str, object],
        history: ChainOfThoughtDAG,
    ) -> Guess:
        g, _ = self.guess_full(turn, setup_answers, history)
        return g

    def guess_full(
        self,
        turn: int,
        setup_answers: Mapping[str, object],
        history: ChainOfThoughtDAG,
    ) -> tuple[Guess, str]:
        """Like guess(), but also returns the raw model output (reasoning +
        guess line) for trajectory capture and CoT->DAG extraction."""
        attribute = self.attribute_schedule[turn % len(self.attribute_schedule)]
        user_prompt = self._build_user_prompt(attribute, setup_answers, history)
        text = ""
        for attempt in range(self.max_retries):
            try:
                text = self._call_api(user_prompt)
            except Exception as exc:  # network / 5xx — retry with backoff
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1.5 ** attempt)
                continue
            parsed = _parse_guess(text, allowed_attributes=[attribute])
            if parsed is not None:
                return parsed, text
            # Parse failure — reinforce the format and retry once.
            user_prompt = (
                user_prompt
                + f"\n\nYour previous output did not match the format. "
                f"End with exactly one line: [{attribute}]: <value>, p=<0.xx>"
            )
        # All retries failed — emit a maximally-uncertain fallback so the
        # pipeline can continue (the score will be ~0, neutral).
        return Guess(attribute=attribute, value="unknown", probability=0.5), text

    def resample(
        self,
        turn: int,
        setup_answers: Mapping[str, object],
        history: ChainOfThoughtDAG,
        k: int = 3,
        temperature: float = 0.9,
    ) -> list[str]:
        """k independent high-temperature answers to the same turn prompt.

        Cached on the Trajectory for consistency-based signals and the
        semantic-entropy / SelfCheckGPT baselines. Parse failures are kept as
        raw text (the baselines cluster text, not parsed guesses).
        """
        attribute = self.attribute_schedule[turn % len(self.attribute_schedule)]
        user_prompt = self._build_user_prompt(attribute, setup_answers, history)
        out = []
        for _ in range(k):
            try:
                out.append(self._call_api(user_prompt, temperature=temperature))
            except Exception:
                continue
        return out

    # ----------------------------------------------------------------- helpers

    def _build_user_prompt(
        self,
        attribute: str,
        setup_answers: Mapping[str, object],
        history: ChainOfThoughtDAG,
    ) -> str:
        setup_str = "\n".join(f"  {k}: {v}" for k, v in setup_answers.items())
        history_str = _format_history(history)
        description = self.attribute_descriptions.get(attribute, "")
        desc_line = f"\nThis attribute means: {description}\n" if description else ""
        return (
            f"Context (one-time, from the user/task):\n{setup_str}\n\n"
            f"Prior turns in this session:\n{history_str}\n\n"
            f"Now produce your guess for attribute: {attribute}{desc_line}\n"
            f"Output format: [{attribute}]: <value>, p=<probability>"
        )

    def _call_api(
        self, user_prompt: str, temperature: Optional[float] = None
    ) -> str:
        resp = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature if temperature is None else temperature,
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self.calls_made += 1
        usage = data.get("usage", {})
        self.input_tokens_used += int(usage.get("prompt_tokens", 0))
        self.output_tokens_used += int(usage.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"]

    # -------------------------------------------------------- budget accounting

    def estimate_cost(self, input_price_per_m: float, output_price_per_m: float) -> float:
        """Rough USD cost given $/M-token rates."""
        return (
            self.input_tokens_used * input_price_per_m / 1_000_000
            + self.output_tokens_used * output_price_per_m / 1_000_000
        )
