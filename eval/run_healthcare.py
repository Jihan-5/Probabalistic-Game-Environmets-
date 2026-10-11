"""Synthetic healthcare main run.

Usage:
    python -u eval/run_healthcare.py --n 200 --model 70B --seed 42

Pipeline:
  1. Sample N healthcare sessions (each = setup + hidden ground truth).
  2. Run LLM agent through PGEProtocol.build_dag() for each.
  3. Label sessions: hallucinating iff agent made >=50% high-confidence wrong guesses.
  4. Train/test split (70/30 by session id).
  5. Fit GraphonEstimator on the *clean* portion of training only.
  6. Score test sessions with -log_likelihood (high = hallucinating).
  7. Bootstrap AUROC CI + Wilcoxon p-value.
  8. Save raw outputs.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc, bootstrap_auroc_ci, wilcoxon_one_sided
from pge.agents.llama_together import (
    LlamaTogetherAgent, MODEL_LLAMA_70B, MODEL_LLAMA_8B, PRICE_PER_M,
)
from pge.benchmarks.synthetic_healthcare import (
    ATTR_DESCRIPTIONS, HEALTHCARE_ATTRIBUTES, HealthcareBehaviorExtractor,
    HealthcareSession, SETUP_QUESTIONS, sample_dataset, session_is_hallucinating,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator
from pge.scoring import bayesian_score


def run_one(
    session: HealthcareSession,
    agent: LlamaTogetherAgent,
    score_gated_edges: bool,
) -> ChainOfThoughtDAG:
    """Build a DAG for one healthcare session."""
    extractor = HealthcareBehaviorExtractor()
    G = ChainOfThoughtDAG()
    for t, attr in enumerate(agent.attribute_schedule):
        guess = agent.guess(t, session.setup_answers, G)
        b = extractor.extract(guess, session.user_state, t)
        s = bayesian_score(guess.probability, b)
        idx = G.add_node(GuessNode(
            attribute=guess.attribute, value=guess.value,
            probability=guess.probability, score=s, turn=t,
        ))
        if idx > 0:
            parent = idx - 1
            if not (score_gated_edges and G.nodes[parent].score <= 0):
                G.add_edge(parent, idx)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--score-gated-edges", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    tag = f"hc_{args.model}_{args.seed}" + ("_sgated" if args.score_gated_edges else "")
    out_path = args.out or f"results/{tag}.npz"
    Path("results").mkdir(exist_ok=True)

    sessions = sample_dataset(args.n, seed=args.seed)
    print(f"[setup] generated {len(sessions)} healthcare sessions (seed={args.seed})", flush=True)

    agent = LlamaTogetherAgent(
        attribute_schedule=HEALTHCARE_ATTRIBUTES,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS,
        temperature=0.2,
        max_tokens=40,
    )

    print(
        f"[run] running {len(sessions)} trajectories on {args.model} "
        f"with {args.concurrency} workers, score_gated_edges={args.score_gated_edges}",
        flush=True,
    )
    t0 = time.time()
    dags: list[ChainOfThoughtDAG | None] = [None] * len(sessions)
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, s, agent, args.score_gated_edges): i for i, s in enumerate(sessions)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                dags[i] = fut.result()
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"  [{i}] ERROR: {type(exc).__name__}: {exc}", flush=True)
            if (completed + failed) % 20 == 0:
                print(f"  progress: {completed + failed}/{len(sessions)} (failed={failed})", flush=True)

    # Pair sessions with their DAGs, dropping failures.
    paired = [(s, d) for s, d in zip(sessions, dags) if d is not None]
    print(f"[run] {len(paired)} trajectories completed in {time.time() - t0:.1f}s", flush=True)

    in_p, out_p = PRICE_PER_M[model]
    print(
        f"[run] agent cost: ${agent.estimate_cost(in_p, out_p):.4f}  "
        f"(in={agent.input_tokens_used}, out={agent.output_tokens_used})",
        flush=True,
    )

    # Label hallucinations from ground-truth.
    labels = np.array([1 if session_is_hallucinating(d, s) else 0 for s, d in paired])
    print(f"[label] hallucinating: {int(labels.sum())}/{len(paired)} = {labels.mean():.1%}", flush=True)

    # Train/test split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(paired))
    n_train = int(len(paired) * args.train_frac)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_dags = [paired[i][1] for i in train_idx]
    train_labels = labels[train_idx]
    test_dags = [paired[i][1] for i in test_idx]
    test_labels = labels[test_idx]

    # Fit graphon on the CLEAN portion of training only.
    clean_train = [d for d, l in zip(train_dags, train_labels) if l == 0]
    print(f"[graphon] fitting on {len(clean_train)} clean training DAGs (B={args.blocks})", flush=True)
    if len(clean_train) < 5:
        print("[graphon] WARNING: very few clean trajectories. Falling back to all-train.", flush=True)
        clean_train = train_dags
    graphon = GraphonEstimator(B=args.blocks).fit(clean_train)

    # Score test DAGs by multiple detection methods.
    #   1) graphon_ll: paper's Eq. 5 (sign-blind, uses |s|*|s| edge weights)
    #   2) neg_cumulative_evidence: -E_T from paper's Eq. 2 (signed, simple)
    #   3) neg_score_fraction: fraction of negative scores (cheapest baseline)
    # For each, higher score -> more hallucinating.
    test_graphon_ll = np.array([graphon.log_likelihood(d) for d in test_dags])
    test_cum_evidence = np.array([sum(n.score for n in d.nodes) for d in test_dags])
    test_neg_frac = np.array([
        sum(1 for n in d.nodes if n.score < 0) / max(1, len(d.nodes))
        for d in test_dags
    ])

    detection_scores = {
        "graphon_ll": -test_graphon_ll,
        "neg_cum_evidence": -test_cum_evidence,
        "neg_score_fraction": test_neg_frac,
    }

    n_test = len(test_dags)
    n_hall = int(test_labels.sum())
    print(f"[eval] test: n={n_test} hallucinating={n_hall} clean={n_test - n_hall}", flush=True)
    if n_hall == 0 or n_hall == n_test:
        print("[eval] degenerate test split, AUROC undefined.", flush=True)
        return

    print()
    print("=" * 70)
    print(f"Healthcare results ({args.model}, n={len(paired)}, seed={args.seed}, "
          f"score_gated={args.score_gated_edges})")
    print("=" * 70)
    print(f"  hallucination rate (overall): {labels.mean():.1%}")
    print()
    print(f"  {'method':28s} {'AUROC':>7s} {'95% CI':>20s} {'p-value':>10s}")
    print(f"  {'-'*68}")
    results_per_method = {}
    for name, scores in detection_scores.items():
        point, lo, hi = bootstrap_auroc_ci(scores, test_labels, n_resamples=1000, seed=args.seed)
        pvalue = wilcoxon_one_sided(scores[test_labels == 1], scores[test_labels == 0])
        results_per_method[name] = (point, lo, hi, pvalue)
        print(f"  {name:28s} {point:7.3f}  [{lo:5.3f}, {hi:5.3f}]  {pvalue:10.4g}")

    np.savez(
        out_path,
        test_labels=test_labels,
        test_graphon_ll=test_graphon_ll,
        test_cum_evidence=test_cum_evidence,
        test_neg_frac=test_neg_frac,
        train_labels=train_labels,
        score_gated_edges=int(args.score_gated_edges),
        agent_input_tokens=agent.input_tokens_used,
        agent_output_tokens=agent.output_tokens_used,
        agent_cost=agent.estimate_cost(in_p, out_p),
        seed=args.seed,
        n_test=n_test,
        n_hall=n_hall,
        **{f"auroc_{k}": v[0] for k, v in results_per_method.items()},
        **{f"auroc_lo_{k}": v[1] for k, v in results_per_method.items()},
        **{f"auroc_hi_{k}": v[2] for k, v in results_per_method.items()},
        **{f"pvalue_{k}": v[3] for k, v in results_per_method.items()},
    )
    print(f"\n[save] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
