"""e5_crossdomain — train on one domain, test on the other (offline, $0).

Reviewer 2AM6's generalization concern: components trained on a benchmark may
overfit it. Here supervised detectors (dag_logreg, gcn if available) are fit
on ALL trajectories of the source domain and evaluated on ALL trajectories of
the target domain (no target supervision at all); zero-shot detectors are
reported per target domain for reference.

Usage: python3 -m eval.e5_crossdomain
Writes results/e5_crossdomain/summary.txt
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import bootstrap_auroc_ci
from pge.baselines.confidence import HighConfContradicted, NegCumulativeEvidence
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.benchmarks import dbbench_v2, healthcare_v2
from pge.trajectory import load_jsonl

try:
    from pge.baselines.gcn import GCNClassifier
except ImportError:  # pragma: no cover
    GCNClassifier = None

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "e5_crossdomain"

DAG_BUILDERS = {
    "dbbench_v2": dbbench_v2.dag_from_trajectory,
    "healthcare_v2": healthcare_v2.dag_from_trajectory,
}


def load_domain(domain: str):
    trajs = []
    for f in sorted(glob.glob(str(REPO / "data" / "trajectories" / domain / "*.jsonl"))):
        trajs.extend(load_jsonl(f))
    dags = [DAG_BUILDERS[domain](t) for t in trajs]
    labels = np.array([t.eval_label for t in trajs])
    return trajs, dags, labels


def main() -> None:
    domains = [d for d in DAG_BUILDERS if (REPO / "data" / "trajectories" / d).exists()]
    data = {d: load_domain(d) for d in domains}
    lines = [
        "# e5_crossdomain — supervised detectors trained on SOURCE, tested on TARGET",
        f"domains: " + ", ".join(f"{d} (n={len(data[d][0])})" for d in domains),
        "",
    ]

    for src in domains:
        for tgt in domains:
            if src == tgt:
                continue
            s_trajs, s_dags, s_y = data[src]
            t_trajs, t_dags, t_y = data[tgt]
            if len(set(t_y)) < 2:
                lines.append(f"{src} -> {tgt}: degenerate target labels, skipped")
                continue
            lines.append(f"## {src} -> {tgt}  (target n={len(t_trajs)}, "
                         f"hall. rate {t_y.mean():.2f})")

            rows = []
            clf = DagFeatureClassifier().fit(s_trajs, s_dags, s_y)
            scores = np.array([clf.score(t, g) for t, g in zip(t_trajs, t_dags)])
            rows.append(("dag_logreg (transferred)", scores))
            if GCNClassifier is not None:
                gcn = GCNClassifier().fit(s_trajs, s_dags, s_y)
                scores = np.array([gcn.score(t, g) for t, g in zip(t_trajs, t_dags)])
                rows.append(("gcn (transferred)", scores))
            for det in (NegCumulativeEvidence(), HighConfContradicted()):
                scores = np.array([det.score(t, g) for t, g in zip(t_trajs, t_dags)])
                rows.append((f"{det.name} (zero-shot ref)", scores))

            for name, scores in rows:
                pt, lo, hi = bootstrap_auroc_ci(scores, t_y, seed=0)
                lines.append(f"  {name:32s} AUROC {pt:.3f}  [{lo:.3f}, {hi:.3f}]")
            lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
