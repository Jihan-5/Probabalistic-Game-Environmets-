"""e7e — task-level (cluster) bootstrap for the dbbench 70B pooled comparison ($0).

dbbench 70B pooled = 70B_{42,123,456}.jsonl: 354 rows, 118 unique tasks,
every task generated 3 times (one per generation seed). The e6 split is by
task_id (all copies on one side), but e6's paired bootstrap resamples ROWS
of the 3 pooled split-seed test sets, treating the 3 generation copies of a
task (and its re-appearance across split seeds' test sets) as independent.
Here the same scores are bootstrapped by resampling UNIQUE task_ids: all
pooled rows of a sampled task (up to 3 gen copies x 3 split seeds) enter
together. Point estimates are identical; only the CIs change.

Detectors as in e6 (GCN skipped): pge_evidence_nb vs selfcheckgpt,
dag_logreg, semantic_entropy (requested), plus highconf_contradicted and
neg_signal_fraction for completeness.

Usage: python3 -m eval.e7_dbbench_cluster
Writes results/e7_dbbench_cluster/summary.txt
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eval.e7_common import diff_cis, fmt_ci, masked, mean_auprc, mean_auroc, run_protocol
from eval.score_trajectories import load_trajectories
from pge.baselines.confidence import HighConfContradicted, NegativeSignalFraction
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.posterior import EvidenceNB
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import SemanticEntropy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7_dbbench_cluster"
# Same file order e6 used (verified: reproduces the e6 item-level CIs exactly).
PATHS = [str(REPO / f"data/trajectories/dbbench_v2/70B_{s}.jsonl") for s in (42, 123, 456)]
FACTORIES = [
    ("pge_evidence_nb", EvidenceNB),
    ("selfcheckgpt", SelfCheckGPT),
    ("dag_logreg", DagFeatureClassifier),
    ("semantic_entropy", SemanticEntropy),
    ("highconf_contradicted", HighConfContradicted),
    ("neg_signal_fraction", NegativeSignalFraction),
]
E6_ITEM = {  # results/e6_posterior/dbbench_70B_pooled.txt, for the reproduction check
    "selfcheckgpt": "+0.015 [-0.022, +0.051]",
    "dag_logreg": "+0.050 [+0.000, +0.101]",
    "semantic_entropy": "+0.054 [+0.009, +0.097]",
    "highconf_contradicted": "+0.198 [+0.141, +0.253]",
    "neg_signal_fraction": "+0.091 [+0.039, +0.141]",
}


def main():
    t0 = time.time()
    trajs = load_trajectories(PATHS)
    labels = np.array([t.eval_label for t in trajs])
    det_trajs = [masked(t) for t in trajs]
    R = run_protocol(det_trajs, labels, FACTORIES)
    y = np.concatenate(R["y"])
    tid = np.concatenate(R["tid"])
    uniq, counts = np.unique(tid, return_counts=True)
    lines = [
        "# e7e dbbench 70B pooled: item-level vs task-level (cluster) bootstrap (offline, $0)",
        "",
        f"n={len(trajs)} rows, {len(set(t.task_id for t in trajs))} unique tasks, hall rate {labels.mean():.3f}.",
        f"Pooled test rows over split seeds 42/123/456: {len(y)}; unique tasks among them: {len(uniq)};",
        f"rows per task in the pooled test set: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}",
        "(3 generation copies x number of split seeds whose test set contains the task).",
        "Item CI = e6 protocol (resample rows). Task CI = resample unique task_ids, all their pooled rows together.",
        "Both: 2000 resamples, rng seed 0, percentile 95% CI. * = excludes 0.",
        "",
        "## mean over split seeds",
    ]
    for n, _ in sorted(FACTORIES, key=lambda f: -mean_auroc(R, f[0])):
        lines.append(f"  {n:22s} AUROC {mean_auroc(R, n):.3f}  AUPRC {mean_auprc(R, n):.3f}   per-seed "
                     + " ".join(f"{x:.3f}" for x in R["auroc"][n]))
    lines += ["", "## pge_evidence_nb - baseline",
              "| baseline | item-level CI (e6) | task-level CI | width item | width task | e6 reported item CI |",
              "|---|---|---|---|---|---|"]
    for b, _ in FACTORIES[1:]:
        item, clus = diff_cis(R, "pge_evidence_nb", b)
        lines.append(f"| {b} | {fmt_ci(item)} | {fmt_ci(clus)} | {item[2] - item[1]:.3f} | {clus[2] - clus[1]:.3f} | "
                     f"{E6_ITEM[b]} |")
    lines += ["", f"runtime {time.time() - t0:.0f}s"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
