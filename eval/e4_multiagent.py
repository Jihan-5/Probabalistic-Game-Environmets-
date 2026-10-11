"""e4_multiagent — multi-agent conflict resolution on healthcare v2. Zero LLM cost.

k=3 heterogeneous agents (a0 grounded, a1 hallucinator, a2 grounded-noisy)
commit independently on n sessions x seeds. Conflicts = attributes where >=2
agents commit different values at p >= 0.7. Resolvers compared on identical
test conflicts (70/30 task split; ground truth used only for scoring and for
fitting best_single_agent on the TRAIN split):

  majority_vote        plurality over all commitments (prob-sum tiebreak)
  confidence_weighted  argmax summed implied_belief(p) (de-biased reports)
  best_single_agent    most accurate agent on TRAIN conflicts, deferred to
  mlcr                 graphon MLCR on the merged cross-agent DAG
                       (graphon fit on ALL training merged DAGs,
                        B=6, feature|abs, normalized)
  oracle               truth if ANY agent proposed it, else majority (ceiling)

Significance: McNemar exact test (binomial on discordant pairs), MLCR vs
majority_vote, per seed and pooled.

Usage:  python3 -m eval.e4_multiagent [--n 500] [--seeds 42 123 456]
Output: results/e4_multiagent/{summary.txt, seed<k>.npz}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from pge.benchmarks.healthcare_v2 import sample_session_v2
from pge.graphon import GraphonEstimator
from pge.multiagent import (
    BestSingleAgent,
    ReliabilityMLCR,
    build_merged_dag,
    collect_commitments,
    confidence_weighted_vote,
    find_conflicts,
    majority_vote,
    mlcr_resolution,
)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "e4_multiagent"

AGENT_PROFILES = [
    {"agent_id": "a0", "hallucinator": False, "noise": 0.05},
    {"agent_id": "a1", "hallucinator": True, "noise": 0.05},
    {"agent_id": "a2", "hallucinator": False, "noise": 0.15},
]
P_MIN = 0.7
TRAIN_FRAC = 0.7
B = 6
METHODS = (
    "majority_vote",
    "confidence_weighted",
    "best_single_agent",
    "mlcr",
    "mlcr_reliability",
    "oracle",
)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes (a vs b).

    Returns (b01, b10, p) with b01 = a wrong & b right, b10 = a right &
    b wrong; p from a two-sided exact binomial on the discordant pairs.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n_disc = b01 + b10
    if n_disc == 0:
        return b01, b10, 1.0
    p = float(binomtest(b01, n_disc, 0.5, alternative="two-sided").pvalue)
    return b01, b10, p


def run_seed(seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    sessions, commitments = [], []
    for i in range(n):
        session = sample_session_v2(rng, i)
        sessions.append(session)
        commitments.append(collect_commitments(session, AGENT_PROFILES, rng))

    split_rng = np.random.default_rng(seed + 1)
    order = split_rng.permutation(n)
    n_train = int(TRAIN_FRAC * n)
    tr, te = order[:n_train], order[n_train:]

    # Graphon fit on ALL training merged DAGs.
    merged = [
        build_merged_dag(sessions[i].schedule, commitments[i]) for i in range(n)
    ]
    graphon = GraphonEstimator(
        B=B, block_method="feature", weighting="abs", normalize=True
    ).fit([merged[i] for i in tr])

    # Supervised resolvers fit on TRAIN conflicts (the only test-blind label use).
    train_conflicts, train_truths, train_items = [], [], []
    for i in tr:
        for cf in find_conflicts(commitments[i], p_min=P_MIN):
            train_conflicts.append(cf)
            train_truths.append(sessions[i].user_state[cf.attribute])
            train_items.append((cf, commitments[i], sessions[i].user_state[cf.attribute]))
    truth_by_id = {id(cf): t for cf, t in zip(train_conflicts, train_truths)}
    bsa = BestSingleAgent().fit(train_conflicts, lambda cf: truth_by_id[id(cf)])
    rmlcr = ReliabilityMLCR().fit(train_items)

    correct: dict[str, list[int]] = {m: [] for m in METHODS}
    for i in te:
        conflicts = find_conflicts(commitments[i], p_min=P_MIN)
        if not conflicts:
            continue
        mlcr_vals = mlcr_resolution(merged[i], conflicts, graphon)
        rmlcr_vals = rmlcr.resolve_session(conflicts, commitments[i])
        for cf in conflicts:
            truth = sessions[i].user_state[cf.attribute]
            proposed = {c.value for c in cf.commitments}
            choices = {
                "majority_vote": majority_vote(cf),
                "confidence_weighted": confidence_weighted_vote(cf),
                "best_single_agent": bsa.predict(cf),
                "mlcr": mlcr_vals[cf.attribute],
                "mlcr_reliability": rmlcr_vals[cf.attribute],
                "oracle": truth if truth in proposed else majority_vote(cf),
            }
            for m in METHODS:
                correct[m].append(int(choices[m] == truth))

    arrs = {m: np.array(v, dtype=int) for m, v in correct.items()}
    b01, b10, p = mcnemar_exact(arrs["mlcr_reliability"], arrs["majority_vote"])
    return {
        "correct": arrs,
        "n_conflicts": len(arrs["majority_vote"]),
        "n_train_conflicts": len(train_conflicts),
        "best_agent": bsa.best_agent_id,
        "mcnemar": (b01, b10, p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# e4_multiagent — n={args.n}, seeds={tuple(args.seeds)}, "
        f"k={len(AGENT_PROFILES)} agents, p_min={P_MIN}, B={B} (feature|abs)",
        "",
    ]
    pooled: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    for seed in args.seeds:
        res = run_seed(seed, args.n)
        arrs = res["correct"]
        for m in METHODS:
            pooled[m].append(arrs[m])
        b01, b10, p = res["mcnemar"]
        lines.append(
            f"## seed {seed}  (test conflicts={res['n_conflicts']}, "
            f"train conflicts={res['n_train_conflicts']}, "
            f"best_single_agent={res['best_agent']})"
        )
        for m in METHODS:
            acc = float(arrs[m].mean()) if len(arrs[m]) else float("nan")
            lines.append(f"  {m:22s} acc {acc:.3f}  ({int(arrs[m].sum())}/{len(arrs[m])})")
        lines.append(
            f"  McNemar mlcr_reliability vs majority: b01(rmlcr wrong,maj right)={b01}, "
            f"b10(rmlcr right,maj wrong)={b10}, p={p:.4f}"
        )
        lines.append("")
        np.savez(
            OUT_DIR / f"seed{seed}.npz",
            **{m: arrs[m] for m in METHODS},
            seed=seed,
            n=args.n,
            mcnemar=np.array(res["mcnemar"]),
        )

    pooled_arrs = {m: np.concatenate(pooled[m]) for m in METHODS}
    b01, b10, p = mcnemar_exact(
        pooled_arrs["mlcr_reliability"], pooled_arrs["majority_vote"]
    )
    n_all = len(pooled_arrs["majority_vote"])
    lines.append(f"## pooled ({n_all} test conflicts across {len(args.seeds)} seeds)")
    for m in METHODS:
        acc = float(pooled_arrs[m].mean()) if n_all else float("nan")
        lines.append(
            f"  {m:22s} acc {acc:.3f}  ({int(pooled_arrs[m].sum())}/{n_all})"
        )
    lines.append(
        f"  McNemar mlcr_reliability vs majority: b01={b01}, b10={b10}, p={p:.4f}"
    )
    if b01 + b10 == 0:
        lines.append(
            "  note: zero discordant pairs — with the implied-belief op_score "
            "prior, mlcr_resolve's graphon term is op-independent (ops never "
            "change adjacency/blocks), so each node keeps its own proposal "
            "whenever implied_belief(p) > 1/k and MLCR reduces to majority "
            "vote on this environment."
        )
    lines.append("")

    report = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
