"""e5_dagsize — detection performance as a function of session DAG size.

Question: do small DAGs undermine graph-level detection?
Bins healthcare-v2 sessions by node count and reports per-bin AUROC for the
leading zero-shot detector (highconf_contradicted) and neg_cum_evidence,
plus the node-count distribution the paper must state explicitly.

Usage: python3 -m eval.e5_dagsize [--paths GLOB]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc, bootstrap_auroc_ci
from pge.baselines.confidence import HighConfContradicted, NegCumulativeEvidence
from pge.trajectory import load_jsonl

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="data/trajectories/healthcare_v2/70B_42.jsonl")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(REPO / args.paths))) or sorted(glob.glob(args.paths))
    trajs = [t for f in files for t in load_jsonl(f)]

    sizes = np.array([len(t.turns) for t in trajs])
    labels = np.array([t.eval_label for t in trajs])
    lines = [
        f"# e5_dagsize  n={len(trajs)} (files={len(files)})",
        f"nodes/session: mean {sizes.mean():.1f} ± {sizes.std():.1f}, "
        f"median {np.median(sizes):.0f}, range {sizes.min()}–{sizes.max()}",
        "",
    ]

    detectors = [HighConfContradicted(), NegCumulativeEvidence()]
    edges = np.quantile(sizes, [0, 1 / 3, 2 / 3, 1.0])
    edges = np.unique(np.round(edges).astype(int))
    lines.append(f"{'bin':>12} {'n':>5} {'hall%':>6}  " +
                 "  ".join(f"{d.name:>22}" for d in detectors))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sizes >= lo) & (sizes <= hi if hi == edges[-1] else sizes < hi)
        y = labels[mask]
        row = f"{f'[{lo},{hi}]':>12} {mask.sum():>5} {100 * y.mean():>5.1f}%  "
        cells = []
        for d in detectors:
            s = np.array([d.score(t) for t, m in zip(trajs, mask) if m])
            if len(set(y)) < 2:
                cells.append(f"{'—':>22}")
                continue
            a = auroc(s, y)
            _, lo_ci, hi_ci = bootstrap_auroc_ci(s, y)
            cells.append(f"{f'{a:.3f} [{lo_ci:.3f},{hi_ci:.3f}]':>22}")
        lines.append(row + "  ".join(cells))

    out_dir = REPO / "results" / "e5_dagsize"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (out_dir / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
