"""e5_robustness — detector degradation under input noise (offline, $0).

Question: how do detectors behave when confidences are
miscalibrated and when the DAG/signal channel is corrupted? We perturb
cached trajectories and re-score:

  conf_jitter   sigma in {0.05, 0.1, 0.2}: Gaussian noise on reported
                probabilities (clipped to [0.01, 0.99])
  signal_flip   rho in {0.1, 0.2}: each nonzero behavior signal flipped
                with prob rho (ON TOP of the environment's own noise)
  edge_dropout  rho in {0.1, 0.2}: each DAG edge dropped with prob rho

Detectors: the zero-shot set + dag_logreg (trained on the CLEAN train split,
scored on perturbed test — the realistic deployment case).

Usage: python3 -m eval.e5_robustness --paths "data/trajectories/*/*.jsonl"
Writes results/e5_robustness/summary.txt
"""

from __future__ import annotations

import argparse
import copy
import glob
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from pge.baselines.confidence import HighConfContradicted, NegCumulativeEvidence
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.structural import ConfidentCascadeScore
from pge.benchmarks import dbbench_v2, healthcare_v2
from pge.trajectory import Trajectory, load_jsonl

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "e5_robustness"

DAG_BUILDERS = {
    "dbbench_v2": dbbench_v2.dag_from_trajectory,
    "healthcare_v2": healthcare_v2.dag_from_trajectory,
}


def perturb(traj: Trajectory, kind: str, level: float, rng) -> Trajectory:
    t = copy.deepcopy(traj)
    if kind == "conf_jitter":
        for turn in t.turns:
            turn.probability = float(
                np.clip(turn.probability + rng.normal(0, level), 0.01, 0.99)
            )
    elif kind == "signal_flip":
        for turn in t.turns:
            for k, v in turn.signals.items():
                if v != 0 and rng.random() < level:
                    turn.signals[k] = -v
    return t  # edge_dropout handled at DAG level


def build_dag(traj: Trajectory, kind: str, level: float, rng):
    G = DAG_BUILDERS[traj.domain](traj)
    if kind == "edge_dropout" and G.edges:
        G.edges = [e for e in G.edges if rng.random() >= level]
    return G


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="data/trajectories/*/*.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / args.paths))) or sorted(glob.glob(args.paths))
    trajs: list[Trajectory] = []
    for f in files:
        trajs.extend(t for t in load_jsonl(f) if t.domain in DAG_BUILDERS)
    if not trajs:
        raise SystemExit("no trajectories found")

    # Task-level split (same convention as the main harness).
    ids = sorted({t.task_id for t in trajs})
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    train_ids = {ids[i] for i in order[: int(args.train_frac * len(ids))]}
    train = [t for t in trajs if t.task_id in train_ids]
    test = [t for t in trajs if t.task_id not in train_ids]
    y = np.array([t.eval_label for t in test])

    clean_dags_train = [DAG_BUILDERS[t.domain](t) for t in train]
    clf = DagFeatureClassifier().fit(train, clean_dags_train, [t.eval_label for t in train])
    zero_shot = [NegCumulativeEvidence(), HighConfContradicted(), ConfidentCascadeScore()]

    conditions = [("clean", 0.0)] + [
        (k, l)
        for k, levels in [
            ("conf_jitter", (0.05, 0.1, 0.2)),
            ("signal_flip", (0.1, 0.2)),
            ("edge_dropout", (0.1, 0.2)),
        ]
        for l in levels
    ]

    lines = [
        f"# e5_robustness  n_test={len(test)} (train={len(train)}, files={len(files)})",
        f"{'condition':22s} " + " ".join(f"{d.name[:14]:>14s}" for d in zero_shot)
        + f" {'dag_logreg':>14s}",
    ]
    for kind, level in conditions:
        kind_id = {"clean": 0, "conf_jitter": 1, "signal_flip": 2, "edge_dropout": 3}[kind]
        prng = np.random.default_rng((args.seed, kind_id, int(level * 100)))
        pert = [perturb(t, kind, level, prng) for t in test]
        dags = [build_dag(t, kind, level, prng) for t in pert]
        row = [f"{kind + (f'({level})' if level else ''):22s}"]
        for det in zero_shot:
            s = np.array([det.score(t, g) for t, g in zip(pert, dags)])
            row.append(f"{auroc(s, y):>14.3f}")
        s = np.array([clf.score(t, g) for t, g in zip(pert, dags)])
        row.append(f"{auroc(s, y):>14.3f}")
        lines.append(" ".join(row))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
