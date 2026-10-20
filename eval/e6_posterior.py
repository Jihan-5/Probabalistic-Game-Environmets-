"""e6_posterior — PGE posterior detectors vs the strongest baselines (offline, $0).

Same protocol as eval/score_trajectories.py: hash split by task_id
(seeds 42/123/456, train_frac 0.7), fit on train, score test once.
Reports AUROC/AUPRC per seed, mean over seeds, and a paired bootstrap CI
on the AUROC difference (candidate minus each baseline) pooled over seeds.

Usage: python3 -m eval.e6_posterior <healthcare|dbbench> <jsonl> [...] [--dag]
Prints the report; redirect to results/e6_posterior/<name>.txt.
"""

from __future__ import annotations

import sys
import time

import numpy as np


from eval.bootstrap_ci import auroc
from eval.score_trajectories import (
    average_precision,
    build_dag,
    load_trajectories,
    split_task_ids,
)
from pge.baselines.confidence import HighConfContradicted, NegativeSignalFraction
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.gcn import GCNClassifier
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import SemanticEntropy
from pge.benchmarks.dbbench_v2 import TURN_DEPENDENCIES
from pge.benchmarks.healthcare_v2 import ATTRIBUTES_V2

from pge.baselines.posterior import DagPosterior, EvidenceNB, IndependentPosterior

SEEDS = (42, 123, 456)


def hc_parents(traj):
    idx = {t.attribute: i for i, t in enumerate(traj.turns)}
    out = []
    for i, t in enumerate(traj.turns):
        deps = ATTRIBUTES_V2.get(t.attribute, ([], []))[1]
        out.append([idx[d[5:]] for d in deps if d.startswith("attr:") and d[5:] in idx and idx[d[5:]] < i])
    return out


def db_parents(traj):
    out = []
    for i, t in enumerate(traj.turns):
        out.append([d for d in TURN_DEPENDENCIES.get(t.attribute, []) if d < i])
    return out


PARENTS = {"healthcare": hc_parents, "dbbench": db_parents}


def make_detectors(domain):
    dets = [
        HighConfContradicted(),
        NegativeSignalFraction(),
        DagFeatureClassifier(),
        GCNClassifier(),
        SelfCheckGPT(),
        SemanticEntropy(),
        IndependentPosterior(),
        IndependentPosterior(by_value=True),
        IndependentPosterior(by_value=True, use_samples=True),
        EvidenceNB(),
    ]
    if "--dag" in sys.argv:
        dets.append(DagPosterior(PARENTS[domain], em_iters=10))
    return dets


def paired_boot_diff(sa, sb, y, n=2000, seed=0):
    """Bootstrap CI for AUROC(sa) - AUROC(sb), resampling items jointly."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() in (0, len(b)):
            continue
        diffs.append(auroc(sa[b], y[b]) - auroc(sb[b], y[b]))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    domain = sys.argv[1]
    paths = [a for a in sys.argv[2:] if not a.startswith("--")]
    trajs = load_trajectories(paths)
    dags = [build_dag(t) for t in trajs]
    labels = np.array([t.eval_label for t in trajs])
    print(f"# {domain}: n={len(trajs)}, unique tasks={len(set(t.task_id for t in trajs))}, "
          f"hall rate={labels.mean():.3f}")

    per_seed = []
    pooled_scores, pooled_y = {}, []
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids([t.task_id for t in trajs], 0.7, seed)
        tr = [i for i, t in enumerate(trajs) if t.task_id in tr_ids]
        te = [i for i, t in enumerate(trajs) if t.task_id in te_ids]
        y_te = labels[te]
        res = {}
        for det in make_detectors(domain):
            t0 = time.time()
            det.fit([trajs[i] for i in tr], [dags[i] for i in tr], labels[tr])
            s = np.array([det.score(trajs[i], dags[i]) for i in te], dtype=float)
            res[det.name] = (auroc(s, y_te), average_precision(s, y_te), s, time.time() - t0)
            extra = f" eps={det.eps}" if hasattr(det, "eps") and det.eps is not None else ""
            print(f"  seed {seed} {det.name:24s} AUROC {res[det.name][0]:.3f}  "
                  f"AUPRC {res[det.name][1]:.3f}  ({res[det.name][3]:.1f}s){extra}", flush=True)
        per_seed.append(res)
        for name, (_, _, s, _) in res.items():
            pooled_scores.setdefault(name, []).append(s)
        pooled_y.append(y_te)

    print("\n## mean over seeds")
    names = list(per_seed[0])
    means = {n: (np.mean([r[n][0] for r in per_seed]), np.mean([r[n][1] for r in per_seed])) for n in names}
    for n, (a, p) in sorted(means.items(), key=lambda kv: -kv[1][0]):
        print(f"  {n:24s} AUROC {a:.3f}  AUPRC {p:.3f}   per-seed "
              + " ".join(f"{r[n][0]:.3f}" for r in per_seed))

    # Paired bootstrap of AUROC differences, pooled across seeds (seed-wise
    # test sets concatenated; each item's scores come from its own seed's fit).
    y_all = np.concatenate(pooled_y)
    cands = [n for n in names if n.startswith("pge_")]
    bases = [n for n in names if not n.startswith("pge_")]
    print("\n## paired bootstrap: candidate - baseline (pooled over seeds)")
    for c in cands:
        sc = np.concatenate(pooled_scores[c])
        for b in bases:
            sb = np.concatenate(pooled_scores[b])
            m, lo, hi = paired_boot_diff(sc, sb, y_all)
            print(f"  {c:22s} - {b:22s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
