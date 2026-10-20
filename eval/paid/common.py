"""Shared plumbing for eval/paid/*: CLI args, client construction, the
offline mock LLM, the parallel runner with checkpointing, and the shared
healthcare-v2 session loop (same session stream, eps channel and DAG
construction as eval/generate_healthcare_v2.py)."""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from pge.agents.paid_client import BudgetExceeded, ChatClient, CostMeter, NonRetryableHTTP
from pge.agents.registry import PROVIDERS, REGISTRY, ModelSpec, resolve
from pge.benchmarks.healthcare_v2 import ATTRIBUTES_V2
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory, load_jsonl, save_jsonl

REPO = Path(__file__).resolve().parent.parent.parent
ALLOWED_VALUES = {a: vals for a, (vals, _) in ATTRIBUTES_V2.items()}

# Agent-side protocol of a healthcare corpus (Trajectory.meta["protocol"]).
#   v2   : the 2026-09 development corpora. The agent saw sign(tanh(F logit p))
#          as feedback (no reaction shown as '+', inverted below p = 0.5).
#          Kept ONLY to replay those corpora exactly.
#   v2fb : confirmatory protocol. The agent sees the raw reaction '+'/'-'/'none'.
_PROMPTS = REPO / "pge" / "prompts"
PROTOCOLS = {
    "v2": {"system_prompt": (_PROMPTS / "healthcare_v2_system.txt").read_text(),
           "feedback_display": "score_sign"},
    "v2fb": {"system_prompt": (_PROMPTS / "healthcare_v2fb_system.txt").read_text(),
             "feedback_display": "signal"},
}


def protocol_of(trajs) -> str:
    """The single agent-side protocol of a corpus (missing meta = 'v2')."""
    ps = {t.meta.get("protocol", "v2") for t in trajs}
    if len(ps) != 1:
        raise SystemExit(f"corpus mixes protocols {sorted(ps)}")
    return ps.pop()


# ------------------------------------------------------------------ CLI


def add_model_args(ap: argparse.ArgumentParser, prefix: str = "", default: str = "70B",
                   help_role: str = "agent") -> None:
    p = f"--{prefix}" if prefix else "--"
    ap.add_argument(f"{p}model", default=default,
                    help=f"{help_role} model: registry short name ({', '.join(REGISTRY)}) or a model id")
    ap.add_argument(f"{p}provider", default=None, choices=sorted(PROVIDERS),
                    help="explicit provider (required for an unregistered model id)")
    ap.add_argument(f"{p}price-in", type=float, default=None, help="$/M input tokens override")
    ap.add_argument(f"{p}price-out", type=float, default=None, help="$/M output tokens override")


def spec_from_args(args, prefix: str = "", short_name: Optional[str] = None) -> ModelSpec:
    g = lambda k: getattr(args, (prefix + k).replace("-", "_"))
    return resolve(g("model"), g("provider"), short_name, g("price_in"), g("price_out"))


def add_run_args(ap: argparse.ArgumentParser, max_cost: float, concurrency: int = 8) -> None:
    ap.add_argument("--max-cost", type=float, default=max_cost,
                    help="HARD cap in USD: no call is sent that could exceed it")
    ap.add_argument("--concurrency", type=int, default=concurrency)
    ap.add_argument("--out", default=None, help="output path override (required with --mock)")
    ap.add_argument("--estimate-only", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.partial.jsonl")
    ap.add_argument("--mock", action="store_true", help="offline dry run against a mock LLM ($0)")
    ap.add_argument("--mock-malformed", type=float, default=0.15,
                    help="mock: fraction of malformed outputs")
    ap.add_argument("--mock-seed", type=int, default=0)


def check_mock_out(args) -> None:
    import os  # PGE_MOCK_ANY_OUT=1: launcher test in a scratch COPY of the repo only
    if args.mock and not args.out and os.environ.get("PGE_MOCK_ANY_OUT") != "1":
        raise SystemExit("--mock requires --out (mock data must never land in data/)")


def make_client(spec: ModelSpec, meter: CostMeter, role: str, args) -> ChatClient:
    transport = MockLLM(args.mock_malformed, args.mock_seed) if args.mock else None
    return ChatClient(spec, meter, role=role, transport=transport)


def preflight(clients: list[ChatClient]) -> None:
    """One 3-token call per distinct endpoint: fail fast on a missing key or
    a model id the provider does not serve."""
    seen = set()
    for c in clients:
        key = (c.spec.provider, c.spec.model_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            c.preflight()
        except NonRetryableHTTP as exc:
            raise SystemExit(f"[preflight] {c.spec.model_id} via {c.spec.provider} rejected: {exc}")
        print(f"[preflight] ok: {c.spec.model_id} via {c.spec.provider}", flush=True)


def print_estimate(label: str, rows: list[tuple[str, ModelSpec, int, float, float]],
                   max_cost: float) -> float:
    """rows: (what, spec, n_calls, est_in_tokens_per_call, est_out_tokens_per_call)."""
    total = 0.0
    print(f"[estimate] {label}")
    for what, spec, n, tin, tout in rows:
        c = spec.cost(int(n * tin), int(n * tout))
        total += c
        print(f"  {what:28s} {spec.model_id} via {spec.provider}: {n} calls x "
              f"(~{tin:.0f} in + ~{tout:.0f} out) @ ${spec.price_in}/${spec.price_out} per M "
              f"[{spec.price_status}] = ${c:.2f}")
    print(f"  TOTAL ~${total:.2f}   hard cap --max-cost ${max_cost:.2f}"
          + ("   (estimate EXCEEDS cap: run will stop at the cap)" if total > max_cost else ""),
          flush=True)
    return total


def ledger_line(experiment: str, config: str, headline: str, cost: float, artifact: str) -> str:
    d = datetime.date.today().isoformat()
    return f"| {d} | {experiment} | {config} | {headline} | ${cost:.2f} | `{artifact}` |"


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_run_json(out: Path, payload: dict) -> Path:
    p = Path(str(out) + ".run.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p


# ------------------------------------------------------------ parallel runner


def run_parallel(items: list, fn: Callable, concurrency: int, meter: CostMeter,
                 on_result: Callable, label: str = "items", progress_every: int = 20,
                 checkpoint: Optional[Callable] = None, checkpoint_every: int = 50) -> dict:
    """Run fn(item) in a thread pool. Stops cleanly when the cost cap trips
    (BudgetExceeded) or on repeated non-retryable HTTP errors. Returns counts."""
    stats = {"done": 0, "failed": 0, "budget_stopped": 0, "cancelled": 0, "errors": {}}
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(fn, it): it for it in items}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
                with lock:
                    on_result(futures[fut], res)
                stats["done"] += 1
            except BudgetExceeded:
                stats["budget_stopped"] += 1
            except CancelledError:
                stats["cancelled"] += 1
            except Exception as exc:
                stats["failed"] += 1
                key = type(exc).__name__
                stats["errors"][key] = stats["errors"].get(key, 0) + 1
                if stats["failed"] <= 5:
                    print(f"  [{label} {futures[fut]!r}] ERROR {key}: {str(exc)[:300]}", flush=True)
                if isinstance(exc, NonRetryableHTTP) and stats["failed"] >= 3 and stats["done"] == 0:
                    print("[abort] repeated non-retryable HTTP errors; stopping", flush=True)
                    meter.stop("repeated non-retryable HTTP errors")
            if meter.tripped:
                for f in futures:
                    f.cancel()
            if n % progress_every == 0:
                print(f"  progress {n}/{len(items)} done={stats['done']} failed={stats['failed']} "
                      f"cost=${meter.cost:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if checkpoint and stats["done"] and stats["done"] % checkpoint_every == 0:
                with lock:
                    checkpoint()
    if stats["budget_stopped"] or meter.tripped:
        print(f"[abort] stopped ({meter.stop_reason}): {stats['done']} {label} completed, "
              f"{stats['failed']} errored, {len(items) - stats['done'] - stats['failed']} not run "
              f"(spent ${meter.cost:.4f} of ${meter.max_cost:.2f})", flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def load_partial(out: Path) -> list[Trajectory]:
    p = Path(str(out) + ".partial.jsonl")
    return load_jsonl(p) if p.exists() else []


def save_partial(trajs: list[Trajectory], out: Path) -> None:
    save_jsonl(sorted(trajs, key=_tid_key), Path(str(out) + ".partial.jsonl"))


def finalize(trajs: list[Trajectory], out: Path) -> int:
    n = save_jsonl(sorted(trajs, key=_tid_key), out)
    p = Path(str(out) + ".partial.jsonl")
    if p.exists():
        p.unlink()
    return n


def _tid_key(t: Trajectory):
    m = re.search(r"(\d+)$", t.task_id)
    return (int(m.group(1)) if m else 0, t.task_id)


# ------------------------------------------------------------ DAG helpers


def add_dag_node(G: ChainOfThoughtDAG, node_idx: dict, attribute: str, value: str,
                 probability: float, signal: int, t: int) -> None:
    """Identical node/edge construction to eval/generate_healthcare_v2.py."""
    idx = G.add_node(GuessNode(attribute=attribute, value=value, probability=probability,
                               score=bayesian_score(probability, signal), turn=t, signal=signal))
    node_idx[attribute] = idx
    for dep in ATTRIBUTES_V2[attribute][1]:
        if dep.startswith("attr:") and dep[5:] in node_idx:
            parent = node_idx[dep[5:]]
            if parent < idx:
                G.add_edge(parent, idx)
    if idx > 0 and (idx - 1, idx) not in G.edges:
        G.add_edge(idx - 1, idx)


def is_failed(status: str) -> bool:
    return status.endswith("failed")


# ------------------------------------------------------------------ mock LLM


class MockLLM:
    """Offline stand-in for both providers. Dispatches on prompt content
    (agent / patient / extractor / CoVe verifier / preflight), returns an
    OpenAI-style response with a usage block. A fraction `malformed` of
    outputs is malformed in realistic ways (markdown, missing brackets,
    percentages, prose with no commitment, invalid values, unparseable
    extractor labels)."""

    is_mock = True

    def __init__(self, malformed: float = 0.15, seed: int = 0):
        self.malformed = malformed
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.calls = 0

    def _r(self) -> float:
        with self._lock:
            return self._rng.random()

    def _choice(self, xs):
        with self._lock:
            return self._rng.choice(xs)

    def __call__(self, url, headers, payload, timeout):
        msgs = payload["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = msgs[-1]["content"]
        if "Reply with the single word OK" in user:
            text = "OK"
        elif "role-playing a patient" in system:
            text = self._patient(system, user)
        elif "label a patient's reaction" in system:
            text = self._extractor(user)
        elif "Do not assume any prior answer is correct" in user:
            text = self._verify(user)
        else:
            text = self._agent(user)
        with self._lock:
            self.calls += 1
        prompt_chars = sum(len(m["content"]) for m in msgs)
        text = text[: payload.get("max_tokens", 200) * 4]
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_chars // 4 + 8, "completion_tokens": len(text) // 4 + 1},
        }

    # ---- roles
    def _allowed(self, user: str, attr: str) -> list[str]:
        return ALLOWED_VALUES.get(attr, ["yes", "no"])

    def _agent(self, user: str) -> str:
        m = re.search(r"Now produce your guess for attribute: (\w+)", user)
        attr = m.group(1) if m else "unknown_attr"
        vals = self._allowed(user, attr)
        v = self._choice(vals)
        stated = re.search(rf'about {attr}: "(.+?)"', user)
        if stated and self._r() < 0.9:  # mock agent uses the patient's answer 90% of the time
            from eval.paid.run_clarification_closed_loop import answer_sentence
            hit = [x for x in vals if answer_sentence(attr, x) == stated.group(1)]
            v = hit[0] if hit else v
        p = round(0.3 + 0.65 * self._r(), 2)
        if "did not match the format" in user and self._r() < 0.7:
            return f"[{attr}]: {v}, p={p}"
        if self._r() >= self.malformed:
            return f"[{attr}]: {v}, p={p}"
        kind = self._choice(["md", "nobracket_pct", "prose", "invalid", "reasoning", "spaced", "empty"])
        return {
            "md": f"**[{attr}]**: {v.capitalize()}, p={p}",
            "nobracket_pct": f"{attr}: {v} (p = {int(p * 100)}%)",
            "prose": "I cannot determine this from the intake answers alone.",
            "invalid": f"[{attr}]: maybe, p=0.5",
            "reasoning": f"The intake suggests this.\nFinal answer:\n[{attr}]: {v}, p={p}",
            "spaced": f"[{attr}]: {v.replace('_', ' ')}, probability: {p}",
            "empty": "",
        }[kind]

    def _patient(self, system: str, user: str) -> str:
        truth = re.search(r"About the topic the assistant just raised: (.+)", system)
        said = re.search(r'The assistant says to you: "(.+)"', user)
        r = self._r()
        if truth is None:  # 'profile' mode: derive truth from the profile text itself
            right = self._profile_right(system, said.group(1) if said else "")
            if right is None:
                return self._choice(["Okay, that sounds fine.", "How long will it take?"])
            truth = None
        else:
            right = bool(said and truth.group(1).strip().rstrip(".") in said.group(1))
        if right:
            if r < 0.6:
                return "Okay, that sounds about right to me."
            if r < 0.72:
                return "Hmm, are you sure about that? I'm not so certain."
            return "Will I be able to drive myself home afterwards?"
        if r < 0.5:
            t = truth.group(1).strip() if truth else "that doesn't fit what I know about myself"
            return f"No, actually I think {t}"
        if r < 0.75:
            return "Okay, if you say so."
        return "How long will the whole appointment take?"

    @staticmethod
    def _profile_right(system: str, said: str):
        """Mock only: rebuild setup/hidden from the profile bullet text and
        compare the statement with derive_ground_truth_v2 (None = unknown)."""
        from pge.benchmarks.healthcare_llmpatient import STATEMENTS, TOPICS
        from pge.benchmarks.healthcare_v2 import derive_ground_truth_v2
        topic = re.search(r"The assistant just raised: (.+?)\. You have not", system)
        attr = next((a for a, t in TOPICS.items() if topic and t == topic.group(1)), None)
        if attr is None:
            return None
        has = lambda x: x in system
        days = re.search(r"treatment recently: (\d+) day", system)
        setup = {
            "age_group": "under_40" if has("under 40") else "over_65" if has("over 65 years") else "40_to_65",
            "recent_treatment": "yes" if days else "no",
            "chronic_conditions": "none" if has("no chronic") else "many" if has("several chronic") else "some",
            "urgency": "low" if has("mild and not urgent") else "high" if has("severe and you were") else "medium",
            "known_allergies": "none" if has("no known allergies") else "multiple" if has("several allergies") else "some",
        }
        hidden = {
            "days_since_treatment": int(days.group(1)) if days else 30,
            "allergy_severity": "severe" if has("reactions have been severe") else "mild",
            "kidney_function": "impaired" if has("do not work as well") else "normal",
            "anxiety_level": "high" if has("very anxious") else "low",
        }
        truth = derive_ground_truth_v2(setup, hidden)[attr]
        matches = [v for v, st in STATEMENTS[attr].items() if st in said]
        if not matches:
            return False  # non-answer
        said_v = max(matches, key=lambda v: len(STATEMENTS[attr][v]))
        return said_v == truth

    def _extractor(self, user: str) -> str:
        reply = user.split("Patient's reply:")[-1].lower()
        if "no, actually" in reply or "are you sure" in reply:
            lab = "CONTRADICTS"
        elif "okay" in reply:
            lab = "AGREES"
        else:
            lab = "NONE"
        if self._r() < self.malformed:
            return self._choice([f"Label: {lab.lower()}", f"The patient {lab.lower()}.", "unclear",
                                 f"{lab}\nThe patient's reply shows this."])
        return lab

    def _verify(self, user: str) -> str:
        m = re.search(r'what is the value of "(\w+)"', user)
        attr = m.group(1) if m else ""
        vals = ALLOWED_VALUES.get(attr, ["yes", "no"])
        if self._r() < self.malformed:
            return "It depends on information not provided."
        return f"[final_answer]: {self._choice(vals)}, p={round(0.4 + 0.5 * self._r(), 2)}"
