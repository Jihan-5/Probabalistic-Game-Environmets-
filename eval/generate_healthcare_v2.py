"""Generate cached healthcare-v2 trajectories with a REAL LLM agent.

The synthetic-agent version of this environment (generate_dataset_v2) is free
and used for Gate G1; this script replaces the synthetic agent with a Together
LLM so the paper's healthcare numbers come from real model behavior. The
noisy-partial feedback channel and eval labels are identical to the synthetic
path: signals = truth-consistency flipped with prob epsilon (0 for
no-feedback attributes), eval label = >=50% of high-confidence guesses wrong
vs NOISE-FREE truth.

Usage:
    python3 -m eval.generate_healthcare_v2 --n 50 --model 70B --seed 42   # pilot
    python3 -m eval.generate_healthcare_v2 --n 600 --model 70B --seed 42  # main

Writes data/trajectories/healthcare_v2/<model>_<seed>.jsonl.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pge.agents.llama_together import (
    MODEL_LLAMA_8B,
    MODEL_LLAMA_70B,
    PRICE_PER_M,
    LlamaTogetherAgent,
)
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2,
    ATTRIBUTES_V2,
    NO_FEEDBACK_ATTRS,
    AgentGuessV2,
    SessionV2,
    eval_label_v2,
    sample_session_v2,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory, Turn, save_jsonl

REPO = Path(__file__).resolve().parent.parent
DATA_OUT = REPO / "data" / "trajectories" / "healthcare_v2"
SYSTEM_PROMPT_HC = (REPO / "pge" / "prompts" / "healthcare_v2_system.txt").read_text()


def run_llm_session(
    session: SessionV2,
    model: str,
    seed: int,
    epsilon: float,
    rng: np.random.Generator,
    agent: LlamaTogetherAgent | None = None,
) -> tuple[Trajectory, LlamaTogetherAgent]:
    """One LLM session. A fresh agent per session (schedules vary); the
    caller aggregates token counts across returned agents.

    `agent` (optional, default None = the original behavior): a pre-built
    agent with the same interface, e.g. eval/paid's provider-routed agent."""
    if agent is None:
        agent = LlamaTogetherAgent(
            attribute_schedule=session.schedule,
            model=model,
            attribute_descriptions=ATTR_DESCRIPTIONS_V2,
            temperature=0.3,
            max_tokens=200,
            system_prompt=SYSTEM_PROMPT_HC,
            request_timeout=45.0,
        )
    G = ChainOfThoughtDAG()
    node_idx: dict[str, int] = {}
    turns: list[Turn] = []
    guesses: dict[str, AgentGuessV2] = {}

    for t, attribute in enumerate(session.schedule):
        guess, raw = agent.guess_full(t, session.setup_answers, G)
        value = str(guess.value).strip().lower()
        guesses[attribute] = AgentGuessV2(attribute, value, guess.probability)

        correct = value == session.user_state[attribute]
        if attribute in NO_FEEDBACK_ATTRS:
            signal = 0
        else:
            signal = 1 if correct else -1
            if rng.random() < epsilon:
                signal = -signal

        turns.append(
            Turn(
                attribute=attribute,
                guess_value=value,
                probability=guess.probability,
                turn=t,
                raw_response=raw,
                signals={"noisy_feedback": signal},
            )
        )
        idx = G.add_node(
            GuessNode(
                attribute=attribute,
                value=value,
                probability=guess.probability,
                score=bayesian_score(guess.probability, signal),
                turn=t,
            )
        )
        node_idx[attribute] = idx
        for dep in ATTRIBUTES_V2[attribute][1]:
            if dep.startswith("attr:") and dep[5:] in node_idx:
                parent = node_idx[dep[5:]]
                if parent < idx:
                    G.add_edge(parent, idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)

    traj = Trajectory(
        task_id=f"hcv2_{session.profile_id}",
        domain="healthcare_v2",
        model=model,
        seed=seed,
        turns=turns,
        eval_label=eval_label_v2(guesses, session),
        label_source="programmatic",
        meta={"epsilon": epsilon, "n_turns": len(session.schedule)},
    )
    return traj, agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=5.0)
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    in_p, out_p = PRICE_PER_M[model]
    rng = np.random.default_rng(args.seed)
    sessions = [sample_session_v2(rng, i) for i in range(args.n)]
    # Per-session signal-noise streams, seeded independently for replay.
    noise_rngs = [np.random.default_rng((args.seed, i)) for i in range(args.n)]

    t0 = time.time()
    results: list[Trajectory] = []
    tokens_in = tokens_out = 0
    failed = 0

    def one(i: int):
        return run_llm_session(sessions[i], model, args.seed, args.epsilon, noise_rngs[i])

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(one, i): i for i in range(args.n)}
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                traj, agent = fut.result()
                results.append(traj)
                tokens_in += agent.input_tokens_used
                tokens_out += agent.output_tokens_used
            except Exception as exc:
                failed += 1
                print(f"  [session {futures[fut]}] ERROR {type(exc).__name__}: {exc}", flush=True)
            cost = tokens_in * in_p / 1e6 + tokens_out * out_p / 1e6
            if done % 20 == 0:
                print(f"  progress {done}/{args.n} failed={failed} cost=${cost:.2f}", flush=True)
            if cost > args.max_cost:
                print("[abort] max-cost reached; cancelling remaining", flush=True)
                for f in futures:
                    f.cancel()
                break

    if not results:
        raise RuntimeError("no sessions completed")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{args.model}_{args.seed}.jsonl"
    n_written = save_jsonl(results, out)
    labels = np.array([t.eval_label for t in results])
    cost = tokens_in * in_p / 1e6 + tokens_out * out_p / 1e6
    print(f"\n[done] {n_written} trajectories -> {out}  ({time.time()-t0:.0f}s)")
    print(f"[labels] hallucinating {labels.sum()}/{len(labels)} = {labels.mean():.1%}")
    print(f"[cost] ${cost:.3f}  in={tokens_in} out={tokens_out}  (${cost/len(results):.4f}/session)")
    print("[ledger] record this run in results/RESULTS.md")


if __name__ == "__main__":
    main()
