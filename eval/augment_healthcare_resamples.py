"""Add final-value resamples to cached healthcare-v2 trajectories.

The sampling-based baselines (semantic entropy, SelfCheckGPT) need k
high-temperature re-answers per turn. The healthcare generation pass did not
collect them; this augmentation REPLAYS each cached turn's exact context
(setup answers + prior-turn history rebuilt from the cache) and requests k
resamples for the UNDERDETERMINED attributes only — the ones where sampling
disagreement is diagnostic. No guesses are altered; the cache is updated
in place (a .bak copy is written first).

Usage: python3 -m eval.augment_healthcare_resamples --model 70B --seed 42 --k 3
"""

from __future__ import annotations

import argparse
import shutil
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
    UNDERDETERMINED,
    dag_from_trajectory,
    sample_session_v2,
)
from pge.trajectory import Trajectory, load_jsonl, save_jsonl

REPO = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_HC = (REPO / "pge" / "prompts" / "healthcare_v2_system.txt").read_text()


def augment_one(traj: Trajectory, session, model: str, k: int) -> LlamaTogetherAgent:
    schedule = [t.attribute for t in traj.turns]
    agent = LlamaTogetherAgent(
        attribute_schedule=schedule,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS_V2,
        temperature=0.3,
        max_tokens=200,
        system_prompt=SYSTEM_PROMPT_HC,
        request_timeout=45.0,
    )
    full_dag = dag_from_trajectory(traj)
    for t, turn in enumerate(traj.turns):
        if turn.attribute not in UNDERDETERMINED or turn.samples:
            continue
        # History visible at turn t = nodes 0..t-1 (prefix of the full DAG).
        prefix = type(full_dag)(
            nodes=full_dag.nodes[:t],
            edges=[(p, c) for p, c in full_dag.edges if c < t],
        )
        turn.samples = agent.resample(
            t, session.setup_answers, prefix, k=k, temperature=0.9
        )
    return agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=600, help="must match generation run")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=4.0)
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    in_p, out_p = PRICE_PER_M[model]
    cache = REPO / "data" / "trajectories" / "healthcare_v2" / f"{args.model}_{args.seed}.jsonl"
    shutil.copy(cache, cache.with_suffix(".jsonl.bak"))
    trajs = load_jsonl(cache)

    # Rebuild the identical session stream for setup answers.
    rng = np.random.default_rng(args.seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(args.n)}

    tokens = [0, 0]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(augment_one, tr, sessions[tr.task_id], model, args.k): tr.task_id
            for tr in trajs
            if tr.task_id in sessions
        }
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                agent = fut.result()
                tokens[0] += agent.input_tokens_used
                tokens[1] += agent.output_tokens_used
            except Exception as exc:
                print(f"  [{futures[fut]}] ERROR {type(exc).__name__}: {exc}", flush=True)
            cost = tokens[0] * in_p / 1e6 + tokens[1] * out_p / 1e6
            if done % 50 == 0:
                print(f"  progress {done}/{len(futures)} cost=${cost:.2f}", flush=True)
            if cost > args.max_cost:
                print("[abort] max-cost reached", flush=True)
                break

    n = save_jsonl(trajs, cache)
    cost = tokens[0] * in_p / 1e6 + tokens[1] * out_p / 1e6
    n_aug = sum(1 for tr in trajs for t in tr.turns if t.samples)
    print(f"\n[done] rewrote {n} trajectories ({n_aug} turns augmented) "
          f"in {time.time()-t0:.0f}s")
    print(f"[cost] ${cost:.3f}  in={tokens[0]} out={tokens[1]}")
    print("[ledger] record this run in results/RESULTS.md")


if __name__ == "__main__":
    main()
