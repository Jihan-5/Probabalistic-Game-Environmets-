"""Provider-routed chat client with a HARD cost cap, plus a robust
commitment parser and a routed healthcare agent (for eval/paid/*).

ADDITIVE: nothing here changes `pge.agents.llama_together`. `RoutedAgent`
subclasses `LlamaTogetherAgent` and inherits its prompt construction
(`_build_user_prompt`, `_format_history`, system prompt handling) unchanged,
so a routed run sends byte-identical prompts. Only transport, accounting and
parsing are new.

Cost cap. Every call first RESERVES its worst-case cost (estimated input
tokens + max_tokens output at the model's price). If spent + in-flight
reservations + this call would exceed the cap, `BudgetExceeded` is raised
and the meter trips: no later call is sent. Actual cost is then settled
from the provider's usage block (or the provider-reported cost, whichever
is larger; if usage is missing the reservation estimate is charged). So the
cap holds even with concurrent workers, up to the accuracy of the prices.

Transport is injectable (`transport(url, headers, payload, timeout) -> dict`)
so every paid script can be dry-run offline against a mock LLM.
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Callable, Optional

from pge import config
from pge.agent import Guess
from pge.agents.llama_together import LlamaTogetherAgent, _parse_guess, _format_history
from pge.agents.registry import ModelSpec


class BudgetExceeded(RuntimeError):
    """Raised before a call that could push spend over --max-cost."""


class NonRetryableHTTP(RuntimeError):
    """4xx that retrying cannot fix (bad model id, auth, no credits)."""


# --------------------------------------------------------------- cost meter


class CostMeter:
    def __init__(self, max_cost: float):
        self.max_cost = float(max_cost)
        self._lock = threading.Lock()
        self.cost = 0.0
        self.reserved = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.usage_missing = 0
        self.tripped = False
        self.stop_reason = ""
        self.by_role: dict[str, dict] = {}

    def stop(self, reason: str) -> None:
        """Refuse all further calls (cost cap, parse-failure guard, ...)."""
        with self._lock:
            if not self.tripped:
                self.stop_reason = reason
            self.tripped = True

    def reserve(self, amount: float) -> None:
        with self._lock:
            if self.tripped:
                raise BudgetExceeded(f"run stopped: {self.stop_reason}")
            if self.cost + self.reserved + amount > self.max_cost:
                self.tripped = True
                self.stop_reason = f"cost cap ${self.max_cost:.2f}"
                raise BudgetExceeded(
                    f"cost cap ${self.max_cost:.2f} reached (spent ${self.cost:.4f}, "
                    f"in flight ${self.reserved:.4f}, next call up to ${amount:.5f})"
                )
            self.reserved += amount

    def release(self, amount: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - amount)

    def settle(self, reserved: float, role: str, model: str, tin: int, tout: int, cost: float,
               usage_missing: bool = False) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.cost += cost
            self.tokens_in += tin
            self.tokens_out += tout
            self.calls += 1
            self.usage_missing += int(usage_missing)
            r = self.by_role.setdefault(
                role, {"model": model, "calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0}
            )
            r["calls"] += 1
            r["tokens_in"] += tin
            r["tokens_out"] += tout
            r["cost"] += cost

    def summary(self) -> dict:
        with self._lock:
            return {
                "cost": round(self.cost, 6),
                "max_cost": self.max_cost,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "calls": self.calls,
                "usage_missing_calls": self.usage_missing,
                "tripped": self.tripped,
                "stop_reason": self.stop_reason,
                "by_role": {k: dict(v, cost=round(v["cost"], 6)) for k, v in self.by_role.items()},
            }


# ---------------------------------------------------------------- transport


def http_transport(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    import requests  # lazy: mock runs never need it

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (400, 401, 402, 403, 404):
        raise NonRetryableHTTP(f"HTTP {resp.status_code} from {url}: {resp.text[:400]}")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
        raise RuntimeError(f"provider error: {str(data['error'])[:400]}")
    return data


Transport = Callable[[str, dict, dict, float], dict]


class ChatClient:
    """One (model, provider) endpoint. Thread-safe; share one per role."""

    def __init__(
        self,
        spec: ModelSpec,
        meter: CostMeter,
        role: str = "agent",
        transport: Optional[Transport] = None,
        api_key: Optional[str] = None,
        timeout: float = 45.0,
        max_http_retries: int = 4,
    ):
        self.spec = spec
        self.meter = meter
        self.role = role
        self.transport = transport or http_transport
        self.mock = bool(getattr(self.transport, "is_mock", False))
        self._api_key = api_key
        self.timeout = timeout
        self.max_http_retries = max_http_retries

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = "mock" if self.mock else config.require(self.spec.key_env)
        return self._api_key

    def chat_full(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> tuple[str, int, int, float]:
        """(text, tokens_in, tokens_out, cost) for one completion."""
        payload = {
            "model": self.spec.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.spec.provider == "openrouter":
            payload["usage"] = {"include": True}  # ask for provider-reported cost
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        est_in = int(sum(len(m["content"]) for m in messages) / 3.0) + 16  # conservative
        worst = self.spec.cost(est_in, max_tokens)

        for attempt in range(self.max_http_retries + 1):
            self.meter.reserve(worst)
            try:
                data = self.transport(self.spec.url, headers, payload, self.timeout)
                text = data["choices"][0]["message"].get("content") or ""
            except NonRetryableHTTP:
                self.meter.release(worst)
                raise
            except Exception as exc:
                if "Timeout" in type(exc).__name__:
                    # The provider may have generated (and billed) before the
                    # client gave up: charge the reservation, never release it.
                    self.meter.settle(worst, self.role, self.spec.model_id, 0, 0, worst,
                                      usage_missing=True)
                else:
                    self.meter.release(worst)
                if attempt == self.max_http_retries:
                    raise
                is_429 = "429" in str(exc)
                time.sleep((4.0 if is_429 else 1.5) * (1.6 ** attempt) + random.random())
                continue
            usage = data.get("usage") or {}
            tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            missing = tin == 0 and tout == 0
            if missing:  # never let a missing usage block disable the cap
                tin, tout = est_in, min(max_tokens, len(text) // 3 + 1)
            cost = self.spec.cost(tin, tout)
            reported = usage.get("cost")
            if isinstance(reported, (int, float)):
                cost = max(cost, float(reported))
            self.meter.settle(worst, self.role, self.spec.model_id, tin, tout, cost, missing)
            return text, tin, tout, cost
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        return self.chat_full(messages, temperature, max_tokens)[0]

    def preflight(self) -> str:
        """One tiny call: catches a missing key or a model id the provider
        does not serve before any real spend."""
        return self.chat(
            [{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0.0,
            max_tokens=3,
        )


# ------------------------------------------------------------ robust parsing

_EDGE = " \t\"'`*.,;:!?()[]{}"
_PROB_RE = r"(?:p|prob|probability|confidence)\s*[:=]\s*(?P<prob>\d*\.?\d+)\s*(?P<pct>%)?"


def normalize_value(value: str, allowed: Optional[list[str]]) -> Optional[str]:
    """Map a raw value onto the allowed set: case, edge punctuation,
    spaces/hyphens vs underscores. None if it is not an allowed value."""
    v = str(value).strip().strip(_EDGE).lower()
    if allowed is None:
        return v
    for cand in (v, re.sub(r"[\s\-]+", "_", v), v.replace("_", " ")):
        if cand in allowed:
            return cand
    return None


def _to_prob(raw: str, pct: Optional[str]) -> Optional[float]:
    try:
        p = float(raw)
    except ValueError:
        return None
    if pct or p > 1.0:
        p = p / 100.0
    return p if 0.0 <= p <= 1.0 else None


def parse_commitment(
    text: str, attribute: str, allowed: Optional[list[str]] = None
) -> tuple[Optional[Guess], str]:
    """(guess, status). Status is one of
      strict                legacy regex (identical to the 70B/8B pipeline)
      strict_invalid_value  legacy regex matched, value outside allowed set
      lenient               markdown / missing brackets / p=70% / 'p: 0.7' ...
      lenient_no_attr       bare 'value, p=0.x' line with an allowed value
      failed                no commitment for THIS attribute (never reassigned)
    """
    g = _parse_guess(text, allowed_attributes=[attribute])
    if g is not None:
        v = normalize_value(g.value, allowed)
        if v is None:
            return Guess(attribute, str(g.value).strip().lower(), g.probability), "strict_invalid_value"
        # legacy pipeline value = raw.strip().lower(); flag when we normalized beyond that
        legacy = str(g.value).strip().lower()
        return Guess(attribute, v, g.probability), ("strict" if v == legacy else "strict_normalized")

    attr_pat = re.escape(attribute).replace("_", "[_ ]")
    line_re = re.compile(
        rf"\[?\s*{attr_pat}\s*\]?\s*[:=\-]+\s*(?P<val>[A-Za-z][A-Za-z_ \-]*?)\s*[,;]?\s*\(?\s*{_PROB_RE}",
        re.IGNORECASE,
    )
    bare_re = re.compile(rf"^\s*(?P<val>[A-Za-z][A-Za-z_ \-]*?)\s*[,;]\s*\(?\s*{_PROB_RE}", re.IGNORECASE)
    lines = [ln.replace("*", "").replace("`", "").strip(" -\t") for ln in text.splitlines()]
    for ln in reversed([ln for ln in lines if ln]):  # final commitment wins
        m = line_re.search(ln)
        if m:
            p = _to_prob(m.group("prob"), m.group("pct"))
            if p is None:
                continue
            v = normalize_value(m.group("val"), allowed)
            if v is not None:
                return Guess(attribute, v, p), "lenient"
    for ln in reversed([ln for ln in lines if ln]):
        m = bare_re.search(ln)
        if m:
            p = _to_prob(m.group("prob"), m.group("pct"))
            v = normalize_value(m.group("val"), allowed)
            if p is not None and v is not None and allowed is not None:
                return Guess(attribute, v, p), "lenient_no_attr"
    return None, "failed"


# ------------------------------------------------------- feedback display

FEEDBACK_DISPLAYS = ("score_sign", "signal")
_SIGNAL_MARK = {1: "+", -1: "-", 0: "none"}


def format_history_signal(history, max_turns: int = 6, reactor: str = "patient") -> str:
    """Prior turns with the RAW feedback signal: '+' / '-' / 'none'.

    The legacy formatter (llama_together._format_history, display
    'score_sign') shows sign(tanh(F * logit p)): 'no reaction' appears as '+'
    and, for p < 0.5, accept and contradict appear inverted. Kept only to
    replay the 2026-09 corpora exactly; every new protocol uses this one.
    Requires GuessNode.signal (set by every builder in this repo).
    `reactor` names who reacts ("patient" for healthcare, the domain's
    USER_ROLE otherwise)."""
    if len(history) == 0:
        return "(no prior guesses)"
    lines = []
    for node in history.nodes[-max_turns:]:
        if node.signal is None:
            raise ValueError("format_history_signal needs GuessNode.signal; build the DAG with signals")
        lines.append(f"  turn {node.turn}: [{node.attribute}]: {node.value}, "
                     f"p={node.probability:.2f}  ({reactor} reaction: {_SIGNAL_MARK[int(node.signal)]})")
    return "\n".join(lines)


def format_history(history, display: str, reactor: str = "patient") -> str:
    if display == "signal":
        return format_history_signal(history, reactor=reactor)
    if display == "score_sign":
        return _format_history(history)
    raise ValueError(f"feedback display must be one of {FEEDBACK_DISPLAYS}, got {display!r}")


# ------------------------------------------------------------- routed agent


class RoutedAgent(LlamaTogetherAgent):
    """LlamaTogetherAgent with provider routing, cost metering and robust
    parsing. Prompts are inherited unchanged.

    After `guess_full`, `last_parse_status` holds the parse status of that
    turn ('retry1_<status>' if the format reminder was needed; 'failed' if no
    attempt parsed, in which case the legacy placeholder
    Guess(value='unknown', p=0.5) is returned and the caller must flag it).
    """

    def __init__(
        self,
        attribute_schedule: list[str],
        client: ChatClient,
        allowed_values: Optional[dict[str, list[str]]] = None,
        feedback_display: str = "score_sign",
        reactor: str = "patient",
        **kwargs,
    ):
        super().__init__(attribute_schedule, model=client.spec.model_id, api_key="(routed)", **kwargs)
        if feedback_display not in FEEDBACK_DISPLAYS:
            raise ValueError(f"feedback_display must be one of {FEEDBACK_DISPLAYS}")
        self.feedback_display = feedback_display
        self.reactor = reactor  # who reacts, in the 'signal' history lines
        self.client = client
        self.base_url = client.spec.url
        self.allowed_values = allowed_values or {}
        self.cost = 0.0
        self.last_parse_status = ""
        self.parse_log: list[str] = []  # one status per guess_full call, in order

    def _build_user_prompt(self, attribute, setup_answers, history):
        """Legacy template, byte-identical for display 'score_sign'; with
        'signal' only the history lines differ."""
        if self.feedback_display == "score_sign":
            return super()._build_user_prompt(attribute, setup_answers, history)
        setup_str = "\n".join(f"  {k}: {v}" for k, v in setup_answers.items())
        description = self.attribute_descriptions.get(attribute, "")
        desc_line = f"\nThis attribute means: {description}\n" if description else ""
        return (
            f"Context (one-time, from the user/task):\n{setup_str}\n\n"
            f"Prior turns in this session:\n{format_history_signal(history, reactor=self.reactor)}\n\n"
            f"Now produce your guess for attribute: {attribute}{desc_line}\n"
            f"Output format: [{attribute}]: <value>, p=<probability>"
        )

    def _call_api(self, user_prompt: str, temperature: Optional[float] = None) -> str:
        text, tin, tout, cost = self.client.chat_full(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        self.calls_made += 1
        self.input_tokens_used += tin
        self.output_tokens_used += tout
        self.cost += cost
        return text

    def guess_full(self, turn, setup_answers, history):
        """Same loop as the legacy agent (max_retries attempts, same format
        reminder). Differences: robust parsing, and an out-of-set value also
        triggers the reminder; if every attempt is out-of-set the model's
        last value is kept and flagged 'strict_invalid_value' (it is wrong
        vs truth by construction, never silently mapped to a valid value)."""
        attribute = self.attribute_schedule[turn % len(self.attribute_schedule)]
        allowed = self.allowed_values.get(attribute)
        user_prompt = self._build_user_prompt(attribute, setup_answers, history)
        text, fallback = "", None
        for attempt in range(self.max_retries):
            text = self._call_api(user_prompt)  # transport retries live in the client
            g, status = parse_commitment(text, attribute, allowed)
            tag = status if attempt == 0 else f"retry{attempt}_{status}"
            if g is not None and status != "strict_invalid_value":
                self._log(tag)
                return g, text
            if g is not None:
                fallback = (tag, g, text)
            # same format reminder as the legacy agent
            user_prompt = (
                user_prompt
                + f"\n\nYour previous output did not match the format. "
                f"End with exactly one line: [{attribute}]: <value>, p=<0.xx>"
            )
        if fallback is not None:
            tag, g, raw = fallback
            self._log(tag)
            return g, raw
        self._log("failed")
        return Guess(attribute=attribute, value="unknown", probability=0.5), text

    def _log(self, status: str) -> None:
        self.last_parse_status = status
        self.parse_log.append(status)

    def resample(self, turn, setup_answers, history, k: int = 3, temperature: float = 0.9):
        """Legacy resample, except BudgetExceeded propagates (the legacy loop
        swallows every exception, which would hide a tripped cap)."""
        attribute = self.attribute_schedule[turn % len(self.attribute_schedule)]
        user_prompt = self._build_user_prompt(attribute, setup_answers, history)
        out = []
        for _ in range(k):
            try:
                out.append(self._call_api(user_prompt, temperature=temperature))
            except (BudgetExceeded, NonRetryableHTTP):
                raise
            except Exception:
                continue
        return out
