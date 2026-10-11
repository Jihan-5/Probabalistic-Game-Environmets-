"""Day 3 main run: fit graphon on train split, compute AUROC on test split.

Usage:
    python -u eval/run_main.py --n 200 --model 70B --seed 42

Pipeline:
  1. Sample N tasks from dbbench dev+standard.
  2. Run real LLM agent on all N tasks -> DAGs + ground-truth labels.
     Labels: hallucinating = NOT extractor.is_correct(dag) (i.e., wrong final answer).
  3. Train/test split (70/30).
  4. Fit GraphonEstimator on the *clean* portion of training DAGs only
     (the paper's protocol: graphon learns "what correct reasoning looks like").
  5. Score test DAGs with -log_likelihood (high = hallucinating).
  6. Report AUROC + 1000-bootstrap CI.
  7. Save raw outputs to .npz so ablations can reuse them.

Run trajectories concurrently with a thread pool — Together's serverless
endpoint handles 8-16 parallel requests fine.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from eval.bootstrap_ci import auroc, bootstrap_auroc_ci, wilcoxon_one_sided
from pge.agents.llama_together import (
    LlamaTogetherAgent, MODEL_LLAMA_70B, MODEL_LLAMA_8B, PRICE_PER_M,
)
from pge.benchmarks.dbbench import (
    DBBENCH_ATTRIBUTES, DbbenchBehaviorExtractor, DbbenchTask, load_dbbench,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator
from pge.scoring import bayesian_score


ATTR_DESCRIPTIONS = {
    "target_column": "the exact column name from the table whose value is the answer to the question. Output one of the column names listed in the context.",
    "match_count": "the number of rows in the table that satisfy the filter condition implied by the question. Output a non-negative integer.",
    "sql_operation": "the primary SQL operation needed to compute the answer (one of: SELECT, WHERE, COUNT, SUM, MIN, MAX, AVG, ORDER BY). Output one operation.",
    "final_answer": "the exact value that answers the question. Output the answer value directly (e.g., a date, number, name).",
}


@dataclass
class TrajectoryRecord:
    task_idx: int
    correct: bool
    log_likelihood: float       # filled in after graphon fit
    final_answer: str
    label: str
    scores: list                # per-turn scores
    adjacency: list             # edges for replay
    block_assignments: list = None


def run_one(task: DbbenchTask, agent: LlamaTogetherAgent) -> tuple[ChainOfThoughtDAG, bool]:
    # Each thread gets its own extractor (SQLite connection is not thread-safe
    # across the bind() boundary). Build one local to this function.
    extractor = DbbenchBehaviorExtractor()
    extractor.bind(task)
    setup = task.setup_answers()
    user_state = task.user_state()
    G = ChainOfThoughtDAG()
    for t, attr in enumerate(agent.attribute_schedule):
        guess = agent.guess(t, setup, G)
        b = extractor.extract(guess, user_state, t)
        s = bayesian_score(guess.probability, b)
        idx = G.add_node(GuessNode(
            attribute=guess.attribute, value=guess.value,
            probability=guess.probability, score=s, turn=t,
        ))
        if idx > 0:
            G.add_edge(idx - 1, idx)
    return G, extractor.is_correct(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--out", default=None, help="Save .npz here (default: results/main_<model>_<seed>.npz)")
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    out_path = args.out or f"results/main_{args.model}_{args.seed}.npz"
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_tasks = load_dbbench([
        "data/raw/dbbench_dev.jsonl",
        "data/raw/dbbench_standard.jsonl",
    ])
    print(f"[setup] loaded {len(all_tasks)} candidate tasks; sampling {args.n}.", flush=True)
    idx = rng.choice(len(all_tasks), size=min(args.n, len(all_tasks)), replace=False)
    tasks = [all_tasks[i] for i in idx]

    agent = LlamaTogetherAgent(
        attribute_schedule=DBBENCH_ATTRIBUTES,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS,
        temperature=0.2,
        max_tokens=80,
    )

    print(f"[run] running {len(tasks)} trajectories on {args.model} with {args.concurrency} workers...", flush=True)
    t0 = time.time()
    dags_and_correct: list[tuple[ChainOfThoughtDAG, bool, int]] = [None] * len(tasks)
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(run_one, task, agent): i for i, task in enumerate(tasks)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                G, correct = fut.result()
                dags_and_correct[i] = (G, correct, int(idx[i]))
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"  [{i}] ERROR: {type(exc).__name__}: {exc}", flush=True)
            if (completed + failed) % 10 == 0:
                print(f"  progress: {completed + failed}/{len(tasks)} (failed={failed})", flush=True)

    # Drop any None entries from failed trajectories.
    dags_and_correct = [x for x in dags_and_correct if x is not None]
    print(f"[run] {len(dags_and_correct)} trajectories completed in {time.time() - t0:.1f}s", flush=True)
    if not dags_and_correct:
        raise RuntimeError("no trajectories completed — abort")

    in_p, out_p = PRICE_PER_M[model]
    print(f"[run] agent cost: ${agent.estimate_cost(in_p, out_p):.4f}  "
          f"(in={agent.input_tokens_used}, out={agent.output_tokens_used})", flush=True)

    # Split train/test.
    n = len(dags_and_correct)
    perm = rng.permutation(n)
    n_train = int(n * args.train_frac)
    train_dags = [dags_and_correct[i][0] for i in perm[:n_train]]
    train_correct = [dags_and_correct[i][1] for i in perm[:n_train]]
    test_dags = [dags_and_correct[i][0] for i in perm[n_train:]]
    test_correct = [dags_and_correct[i][1] for i in perm[n_train:]]

    # Fit graphon on the CORRECT portion of training only (paper's protocol:
    # learn what clean reasoning looks like).
    clean_train = [d for d, c in zip(train_dags, train_correct) if c]
    if len(clean_train) < 5:
        print(f"[graphon] WARNING: only {len(clean_train)} correct training DAGs — graphon will be noisy", flush=True)
        clean_train = train_dags  # fall back to all training data
    print(f"[graphon] fitting on {len(clean_train)} clean training DAGs (B={args.blocks})", flush=True)
    graphon = GraphonEstimator(B=args.blocks).fit(clean_train)

    # Score test DAGs.
    test_ll = np.array([graphon.log_likelihood(d) for d in test_dags])
    test_labels = np.array([0 if c else 1 for c in test_correct])  # 1 = hallucinating
    test_scores = -test_ll  # higher score -> more hallucinating

    n_test = len(test_dags)
    n_hall = int(test_labels.sum())
    print(f"[eval] test set: n={n_test}  hallucinating={n_hall}  clean={n_test - n_hall}", flush=True)
    if n_hall == 0 or n_hall == n_test:
        print(f"[eval] degenerate split — cannot compute AUROC", flush=True)
        return

    point, lo, hi = bootstrap_auroc_ci(test_scores, test_labels, n_resamples=1000, seed=args.seed)
    pvalue = wilcoxon_one_sided(test_scores[test_labels == 1], test_scores[test_labels == 0])

    print()
    print("=" * 70)
    print(f"Main results ({args.model}, n={n}, seed={args.seed})")
    print("=" * 70)
    print(f"  train correctness: {sum(train_correct)}/{len(train_correct)} = {sum(train_correct)/len(train_correct):.1%}")
    print(f"  test correctness:  {sum(test_correct)}/{n_test} = {sum(test_correct)/n_test:.1%}")
    print(f"  AUROC (test set): {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  Wilcoxon (one-sided) p-value: {pvalue:.4g}")
    print(f"  clean mean log-lik:  {test_ll[test_labels == 0].mean():.3f}")
    print(f"  hall. mean log-lik:  {test_ll[test_labels == 1].mean():.3f}")

    # Save raw outputs for downstream ablations and figures.
    np.savez(
        out_path,
        test_scores=test_scores,
        test_labels=test_labels,
        test_ll=test_ll,
        train_correct=np.array(train_correct),
        test_correct=np.array(test_correct),
        auroc_point=point,
        auroc_lo=lo,
        auroc_hi=hi,
        wilcoxon_p=pvalue,
        agent_input_tokens=agent.input_tokens_used,
        agent_output_tokens=agent.output_tokens_used,
        agent_cost=agent.estimate_cost(in_p, out_p),
        model=str(model),
        seed=args.seed,
    )
    print(f"[save] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
