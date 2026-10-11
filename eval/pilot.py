"""Day 1 pilot: 10 dbbench trajectories end-to-end on cheap 8B model.

Goal: confirm wiring, prompts produce valid guesses, behavior extractor
returns non-degenerate signals, judge produces sensible labels. Does NOT
spend the main-run budget.

Kill criteria (per plan):
  - pilot AUROC < 0.55 -> stop, reassess
  - judge labels look random -> stop, reassess
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from pge.agents.llama_together import (
    LlamaTogetherAgent, MODEL_LLAMA_70B, MODEL_LLAMA_8B, PRICE_PER_M,
)
from pge.benchmarks.dbbench import (
    DBBENCH_ATTRIBUTES, DbbenchBehaviorExtractor, DbbenchTask, load_dbbench,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.judges.llm_judge import LLMJudge
from pge.scoring import bayesian_score


ATTR_DESCRIPTIONS = {
    "target_column": "the exact column name from the table whose value is the answer to the question. Output one of the column names listed in the context.",
    "match_count": "the number of rows in the table that satisfy the filter condition implied by the question. Output a non-negative integer.",
    "sql_operation": "the primary SQL operation needed to compute the answer (one of: SELECT, WHERE, COUNT, SUM, MIN, MAX, AVG, ORDER BY). Output one operation.",
    "final_answer": "the exact value that answers the question. Output the answer value directly (e.g., a date, number, name).",
}


def run_one_trajectory(
    task: DbbenchTask,
    agent: LlamaTogetherAgent,
    extractor: DbbenchBehaviorExtractor,
) -> tuple[ChainOfThoughtDAG, bool]:
    """Run one task end-to-end, return (DAG, agent_got_correct_final_answer)."""
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
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="8B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip LLM judge (saves a few cents per task)")
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    rng = np.random.default_rng(args.seed)

    all_tasks = load_dbbench([
        "data/raw/dbbench_dev.jsonl",
        "data/raw/dbbench_standard.jsonl",
    ])
    print(f"Loaded {len(all_tasks)} candidate tasks; sampling {args.n}.")
    idx = rng.choice(len(all_tasks), size=args.n, replace=False)
    tasks = [all_tasks[i] for i in idx]

    agent = LlamaTogetherAgent(
        attribute_schedule=DBBENCH_ATTRIBUTES,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS,
        temperature=0.2,
        max_tokens=80,
    )
    extractor = DbbenchBehaviorExtractor()
    judge = None if args.skip_judge else LLMJudge()

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        try:
            G, correct = run_one_trajectory(task, agent, extractor)
        except Exception as exc:
            print(f"  [{i}] ERROR: {type(exc).__name__}: {exc}")
            continue
        # Score summary
        scores = [n.score for n in G.nodes]
        guess_vals = [(n.attribute, n.value, n.probability, n.score) for n in G.nodes]
        results.append({
            "task_idx": int(idx[i]),
            "correct": correct,
            "mean_score": float(np.mean(scores)),
            "n_pos": int(sum(1 for s in scores if s > 0)),
            "n_neg": int(sum(1 for s in scores if s < 0)),
            "final_answer": next((n.value for n in G.nodes if n.attribute == "final_answer"), None),
            "label": task.label,
        })
        print(
            f"  [{i:2d}] correct={int(correct)}  "
            f"+/-={results[-1]['n_pos']}/{results[-1]['n_neg']}  "
            f"final={results[-1]['final_answer']!r:50s} "
            f"label={task.label!r}"
        )

    elapsed = time.time() - t0
    in_p, out_p = PRICE_PER_M[model]
    agent_cost = agent.estimate_cost(in_p, out_p)

    n_correct = sum(r["correct"] for r in results)
    print()
    print("=" * 70)
    print(f"Pilot summary ({args.model}, n={len(results)}, {elapsed:.1f}s)")
    print("=" * 70)
    print(f"  agent: in={agent.input_tokens_used}  out={agent.output_tokens_used}  cost=${agent_cost:.4f}")
    print(f"  task correctness: {n_correct}/{len(results)} = {n_correct/max(1,len(results)):.1%}")
    if judge is not None:
        # Run judge on a sample
        labels = []
        for task, r in zip(tasks[:len(results)], results):
            # We need the DAG; re-run not needed because we already discarded it.
            # Instead, judge with the recorded final answer.
            from pge.dag import ChainOfThoughtDAG, GuessNode
            mini_dag = ChainOfThoughtDAG()
            mini_dag.add_node(GuessNode(
                attribute="final_answer", value=r["final_answer"] or "unknown",
                probability=0.7, score=r["mean_score"], turn=0,
            ))
            jr = judge.label(task.setup_answers(), task.user_state(), mini_dag)
            labels.append(jr.hallucinating)
        n_hall = sum(labels)
        print(f"  judge cost: ${judge.estimate_cost():.4f}")
        print(f"  judge hallucination rate: {n_hall}/{len(labels)} = {n_hall/max(1,len(labels)):.1%}")
        # Agreement between ground-truth-correctness and judge-says-not-hallucinating:
        agree = sum(1 for r, l in zip(results, labels) if (not l) == r["correct"])
        print(f"  judge<->correctness agreement: {agree}/{len(labels)} = {agree/max(1,len(labels)):.1%}")


if __name__ == "__main__":
    main()
