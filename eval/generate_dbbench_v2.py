"""Generate cached dbbench-v2 trajectories (the only paid step for this domain).

Usage:
    python3 -m eval.generate_dbbench_v2 --n 50 --model 70B --seed 42   # pilot
    python3 -m eval.generate_dbbench_v2 --n 700 --model 70B --seed 42  # main

Writes data/trajectories/dbbench_v2/<model>_<seed>.jsonl plus a cost line to
stdout. All detector work then runs offline against the JSONL cache
(eval/score_trajectories.py). Pilot gate (plan Phase 2): scale up only if
task accuracy lands in ~30-70% (need both classes populated) and cost/task
implies total domain cost stays within the $20 generation allocation.
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
from pge.benchmarks.dbbench import load_dbbench
from pge.benchmarks.dbbench_v2 import (
    ATTR_DESCRIPTIONS_V2,
    DBBENCH_V2_ATTRIBUTES,
    filter_tasks_v2,
    run_session_v2,
)
from pge.trajectory import save_jsonl

REPO = Path(__file__).resolve().parent.parent
DATA_OUT = REPO / "data" / "trajectories" / "dbbench_v2"
SYSTEM_PROMPT_V2 = (REPO / "pge" / "prompts" / "dbbench_v2_system.txt").read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--k-resamples", type=int, default=3)
    ap.add_argument("--max-cost", type=float, default=5.0,
                    help="Abort submitting new sessions past this USD cost.")
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    in_p, out_p = PRICE_PER_M[model]
    rng = np.random.default_rng(args.seed)

    tasks_all = filter_tasks_v2(
        load_dbbench(
            [REPO / "data/raw/dbbench_dev.jsonl", REPO / "data/raw/dbbench_standard.jsonl"]
        )
    )
    print(f"[setup] {len(tasks_all)} v2-eligible tasks; sampling {args.n}", flush=True)
    idx = rng.choice(len(tasks_all), size=min(args.n, len(tasks_all)), replace=False)
    tasks = [(int(i), tasks_all[i]) for i in idx]

    agent = LlamaTogetherAgent(
        attribute_schedule=DBBENCH_V2_ATTRIBUTES,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS_V2,
        temperature=0.2,
        max_tokens=300,  # room for THOUGHT lines + commitment
        system_prompt=SYSTEM_PROMPT_V2,
        request_timeout=45.0,
    )

    t0 = time.time()
    results = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(run_session_v2, task, agent, ti, args.seed, args.k_resamples): ti
            for ti, task in tasks
        }
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                traj, _ = fut.result()
                results.append(traj)
            except Exception as exc:
                failed += 1
                print(f"  [task {futures[fut]}] ERROR {type(exc).__name__}: {exc}", flush=True)
            if done % 10 == 0:
                cost = agent.estimate_cost(in_p, out_p)
                print(
                    f"  progress {done}/{len(tasks)} failed={failed} cost=${cost:.2f}",
                    flush=True,
                )
                if cost > args.max_cost:
                    print("[abort] max-cost reached; cancelling remaining", flush=True)
                    for f in futures:
                        f.cancel()
                    break

    if not results:
        raise RuntimeError("no trajectories completed")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{args.model}_{args.seed}.jsonl"
    n_written = save_jsonl(results, out)

    labels = np.array([t.eval_label for t in results])
    cost = agent.estimate_cost(in_p, out_p)
    print(f"\n[done] {n_written} trajectories -> {out}  ({time.time()-t0:.0f}s)")
    print(f"[accuracy] task correct: {(labels == 0).mean():.1%}  "
          f"(hallucinating {(labels == 1).sum()}/{len(labels)})")
    print(f"[cost] ${cost:.3f}  in={agent.input_tokens_used} out={agent.output_tokens_used} "
          f"calls={agent.calls_made}  (${cost/len(results):.4f}/task)")
    print("[ledger] record this run in results/RESULTS.md")


if __name__ == "__main__":
    main()
