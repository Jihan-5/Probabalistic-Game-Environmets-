"""e7a — label-definition sensitivity (healthcare 70B and 8B, offline, $0).

Labels are RECOMPUTED from reconstructed per-turn truth (analysis only) for
confidence threshold high in {0.5, 0.6, 0.7} x error fraction frac in
{0.3, 0.5, 0.7}; same rule as eval_label_v2 otherwise. (0.6, 0.5) is the
original eval_label and must reproduce e6 exactly (checked below).
Definitions with < 5% or > 95% positives are skipped and reported.

Detectors per definition (all fit on the train split of THAT label):
  V3  IndependentPosterior(by_value=True, high=high, frac=frac)
  V1  IndependentPosterior(high=high, frac=frac)
  HighConfContradicted(threshold=high), NegativeSignalFraction,
  DagFeatureClassifier (retrained per label), SelfCheckGPT.
Fixed a priori, nothing tuned: V3's confidence-bucket edges stay at
posterior.BUCKETS (0.6/0.65/0.75) for every definition; dag_logreg's
internal high-confidence feature stays at 0.6.

Usage: python3 -m eval.e7_label_sensitivity [70B] [8B]
Writes results/e7_label_sensitivity/summary.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from eval.e7_common import (
    build_dag, diff_cis, fmt_ci, label_from_truth, load_corpus, masked,
    mean_auprc, mean_auroc, run_protocol,
)
from pge.baselines.confidence import HighConfContradicted, NegativeSignalFraction
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.posterior import IndependentPosterior
from pge.baselines.selfcheckgpt import SelfCheckGPT

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7_label_sensitivity"
HIGHS = (0.5, 0.6, 0.7)
FRACS = (0.3, 0.5, 0.7)
BASELINES = ("highconf_contradicted", "neg_signal_fraction", "dag_logreg", "selfcheckgpt")


def factories(high, frac):
    return [
        ("V3", lambda: IndependentPosterior(by_value=True, high=high, frac=frac)),
        ("V1", lambda: IndependentPosterior(high=high, frac=frac)),
        ("highconf_contradicted", lambda: HighConfContradicted(threshold=high)),
        ("neg_signal_fraction", NegativeSignalFraction),
        ("dag_logreg", DagFeatureClassifier),
        ("selfcheckgpt", SelfCheckGPT),
    ]


def run_corpus(tag, lines):
    trajs, truth = load_corpus(tag)
    det_trajs = [masked(t) for t in trajs]  # detectors never see eval_label
    dags = [build_dag(t) for t in det_trajs]
    orig = np.array([t.eval_label for t in trajs])
    lines.append(f"## healthcare {tag} (n={len(trajs)})")
    lines.append("")
    lines.append("| high | frac | hall rate | V3 | V1 | HCC(high) | NSF | dag_logreg | SelfCheck | best baseline | V3 - best (item CI) | V3 - best (task CI) | V3 - V1 (item CI) | V3 AUPRC | best-baseline AUPRC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for high in HIGHS:
        for frac in FRACS:
            y = np.array([label_from_truth(t, w, high, frac) for t, w in zip(trajs, truth)])
            if (high, frac) == (0.6, 0.5):
                assert (y == orig).all(), "recomputed label != eval_label at (0.6, 0.5)"
            rate = y.mean()
            if rate < 0.05 or rate > 0.95:
                lines.append(f"| {high} | {frac} | {rate:.3f} | SKIPPED (<5% or >95% positives) |||||||||||||")
                rows.append((high, frac, rate, None))
                continue
            t0 = time.time()
            R = run_protocol(det_trajs, y, factories(high, frac), dags=dags)
            best = max(BASELINES, key=lambda b: mean_auroc(R, b))
            item, clus = diff_cis(R, "V3", best)
            v1item, _ = diff_cis(R, "V3", "V1", cluster=False)
            cells = " | ".join(f"{mean_auroc(R, n):.3f}" for n in
                               ("V3", "V1", "highconf_contradicted", "neg_signal_fraction", "dag_logreg", "selfcheckgpt"))
            lines.append(f"| {high} | {frac} | {rate:.3f} | {cells} | {best} | {fmt_ci(item)} | {fmt_ci(clus)} | "
                         f"{fmt_ci(v1item)} | {mean_auprc(R, 'V3'):.3f} | {mean_auprc(R, best):.3f} |")
            rows.append((high, frac, rate, (R, best, item, clus, time.time() - t0)))
            print(f"[{tag}] high={high} frac={frac} rate={rate:.3f} V3={mean_auroc(R, 'V3'):.3f} "
                  f"best={best} {mean_auroc(R, best):.3f} diff={fmt_ci(item)} ({time.time() - t0:.1f}s)", flush=True)
    lines.append("")
    lines.append("Per-seed AUROC (seeds 42/123/456) and learned eps for V3:")
    for high, frac, rate, r in rows:
        if r is None:
            continue
        R, best, *_ = r
        per = "  ".join(f"{n} " + "/".join(f"{x:.3f}" for x in R["auroc"][n])
                        for n in ("V3", "V1", best))
        lines.append(f"  ({high}, {frac}): {per}   V3 eps {R['eps']['V3']}")
    n_eval = sum(r is not None for *_, r in rows)
    wins_sig = sum(1 for *_, r in rows if r is not None and r[2][1] > 0)
    wins_pt = sum(1 for *_, r in rows if r is not None and r[2][0] > 0)
    lines.append("")
    lines.append(f"V3 beats the best baseline (point) on {wins_pt}/{n_eval} evaluable definitions; "
                 f"item-level CI excludes 0 on {wins_sig}/{n_eval}.")
    lines.append("")
    return rows


def main():
    tags = [a for a in sys.argv[1:] if a in ("70B", "8B")] or ["70B", "8B"]
    t0 = time.time()
    lines = [
        "# e7a label-definition sensitivity (offline, $0)",
        "",
        "Labels recomputed from RECONSTRUCTED per-turn truth (analysis only) with the eval_label_v2 rule:",
        "label = 1 iff (#wrong among commitments with p >= high) / (#commitments with p >= high) >= frac;",
        "0 if no commitment has p >= high. Detectors see the original recorded feedback channel.",
        "Protocol = e6_posterior: split_task_ids(ids, 0.7, seed), seeds 42/123/456, fit on train, test once.",
        "AUROC columns = mean over seeds. 'best baseline' = highest mean AUROC among",
        "{HighConfContradicted(threshold=high), NegativeSignalFraction, dag_logreg, SelfCheckGPT}, chosen post hoc",
        "(favours the baselines). CIs: paired bootstrap (2000) of AUROC differences on the 3 test sets pooled;",
        "'item' = e6 protocol (rows resampled), 'task' = resample unique task_ids. * = CI excludes 0.",
        "Fixed a priori: V3 bucket edges 0.6/0.65/0.75 (not re-aligned to high), shrinkage k=5, eps grid 0.05-0.45.",
        "",
    ]
    for tag in tags:
        run_corpus(tag, lines)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    name = "summary.txt" if len(tags) == 2 else f"summary_{tags[0]}.txt"
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
