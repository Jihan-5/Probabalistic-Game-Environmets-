"""e6_robustness — robustness of the value-conditioned posterior (offline, $0).

1. Fresh split seeds never used during design (7, 2024, 31337, 99, 555).
2. Sensitivity to the one fixed hyperparameter (shrinkage strength k).
3. The repo's perturbation suite: fit on the CLEAN train split, score a
   perturbed test split (confidence jitter, extra signal flips).

Usage: python3 -m eval.e6_robustness <jsonl>
"""

from __future__ import annotations

import sys

import numpy as np


from eval.bootstrap_ci import auroc
from eval.e5_robustness import perturb
from eval.score_trajectories import build_dag, load_trajectories, split_task_ids
from pge.baselines.confidence import HighConfContradicted
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.selfcheckgpt import SelfCheckGPT

from pge.baselines.posterior import IndependentPosterior


def split(trajs, seed):
    tr_ids, te_ids = split_task_ids([t.task_id for t in trajs], 0.7, seed)
    tr = [i for i, t in enumerate(trajs) if t.task_id in tr_ids]
    te = [i for i, t in enumerate(trajs) if t.task_id in te_ids]
    return tr, te


def fit_score(det, trajs, dags, labels, tr, te_trajs, te_dags):
    det.fit([trajs[i] for i in tr], [dags[i] for i in tr], labels[tr])
    return np.array([det.score(t, g) for t, g in zip(te_trajs, te_dags)])


def main():
    trajs = load_trajectories(sys.argv[1:])
    dags = [build_dag(t) for t in trajs]
    labels = np.array([t.eval_label for t in trajs])

    def dets():
        return [IndependentPosterior(by_value=True), HighConfContradicted(),
                DagFeatureClassifier(), SelfCheckGPT()]

    print("## 1. fresh split seeds (never used in design)")
    fresh = (7, 2024, 31337, 99, 555)
    table = {}
    for seed in fresh:
        tr, te = split(trajs, seed)
        for det in dets():
            s = fit_score(det, trajs, dags, labels, tr, [trajs[i] for i in te], [dags[i] for i in te])
            table.setdefault(det.name, []).append(auroc(s, labels[te]))
    for name, v in sorted(table.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {name:22s} mean {np.mean(v):.3f}  per-seed " + " ".join(f"{x:.3f}" for x in v))
    wins = sum(a > max(table[b][i] for b in table if b != "pge_posterior_value")
               for i, a in enumerate(table["pge_posterior_value"]))
    print(f"  pge_posterior_value best on {wins}/{len(fresh)} fresh seeds")

    print("\n## 2. shrinkage strength k (fixed at 5 a priori; sensitivity only)")
    for k in (1, 2, 5, 10, 20, 50):
        v = []
        for seed in (42, 123, 456):
            tr, te = split(trajs, seed)
            det = IndependentPosterior(prior_strength=k, by_value=True)
            s = fit_score(det, trajs, dags, labels, tr, [trajs[i] for i in te], [dags[i] for i in te])
            v.append(auroc(s, labels[te]))
        print(f"  k={k:<3d} mean AUROC {np.mean(v):.3f}")

    print("\n## 3. perturbation suite (fit clean train, score perturbed test; seeds 42/123/456)")
    for kind, level in (("none", 0), ("conf_jitter", 0.05), ("conf_jitter", 0.1), ("conf_jitter", 0.2),
                        ("signal_flip", 0.1), ("signal_flip", 0.2)):
        res = {}
        for seed in (42, 123, 456):
            tr, te = split(trajs, seed)
            rng = np.random.default_rng(seed)
            te_trajs = [trajs[i] if kind == "none" else perturb(trajs[i], kind, level, rng) for i in te]
            te_dags = [build_dag(t) for t in te_trajs]
            for det in dets():
                s = fit_score(det, trajs, dags, labels, tr, te_trajs, te_dags)
                res.setdefault(det.name, []).append(auroc(s, labels[te]))
        line = "  ".join(f"{n} {np.mean(v):.3f}" for n, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])))
        print(f"  {kind:11s} {level:<4}  {line}")


if __name__ == "__main__":
    main()
