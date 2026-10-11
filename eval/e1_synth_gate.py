"""e1_synth_gate — Gate G1 go/no-go for the rebuilt core method. Zero LLM cost.

Healthcare v2 (non-circular noisy feedback, dependency DAGs, error cascades),
n sessions x 3 seeds, mixed grounded/hallucinating agent population.
Detectors compared on identical cached data:

  Graphon variants (fit on CLEAN training sessions only, score = -loglik):
    position|abs        — the ORIGINAL estimator (honest ablation arm)
    degree|abs, spectral|abs
    feature|abs, feature|signed  — the rebuilt candidates

  Supervised:  dag_logreg (full PGE minus graphon — the fair ablation)
  Zero-shot:   mean_confidence, neg_cum_evidence, neg_signal_fraction,
               highconf_contradicted

GATE CRITERIA (from the approved plan):
  GO          best graphon variant AUROC >= 0.65 AND beats every zero-shot
              confidence baseline by >= 0.05 with non-overlapping 95% CIs,
              on >= 2 of 3 seeds.
  CONDITIONAL graphon ~ dag_logreg (within CI): structure helps but graphon
              is "a detector", not "the detector" — detector-zoo framing.
  NO-GO       nothing structural beats the confidence baselines.

Usage:  python3 -m eval.e1_synth_gate [--n 1000]
Output: results/e1_synth_gate/{summary.txt, seed<k>.npz}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import bootstrap_auroc_ci
from pge.baselines.confidence import (
    HighConfContradicted,
    MeanConfidence,
    NegCumulativeEvidence,
    NegativeSignalFraction,
)
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.structural import (
    CascadeLift,
    ConfidentCascadeScore,
    NegNegEdgeFraction,
)
from pge.benchmarks.healthcare_v2 import generate_dataset_v2
from pge.graphon import GraphonEstimator

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "e1_synth_gate"

SEEDS = (42, 123, 456)
GRAPHON_VARIANTS = [
    ("position", "abs"),
    ("degree", "abs"),
    ("spectral", "abs"),
    ("feature", "abs"),
    ("feature", "signed"),
]
ZERO_SHOT_BASELINES = (
    MeanConfidence,
    NegCumulativeEvidence,
    NegativeSignalFraction,
    HighConfContradicted,
)
# Zero-shot structural statistics: the direct test of "does structure add
# anything beyond per-node marginals" — scored but not part of the gate's
# baseline set (they are PGE-family candidates, not baselines).
STRUCTURAL_DETECTORS = (NegNegEdgeFraction, CascadeLift, ConfidentCascadeScore)
B = 6
TRAIN_FRAC = 0.7


def run_seed(seed: int, n: int) -> dict[str, tuple[float, float, float]]:
    trajs, dags = generate_dataset_v2(n, seed=seed)
    labels = np.array([t.eval_label for t in trajs])

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(TRAIN_FRAC * n)
    tr, te = order[:n_train], order[n_train:]
    y_te = labels[te]

    results: dict[str, tuple[float, float, float]] = {}

    # Graphon variants: fit on clean training DAGs only.
    clean_tr_dags = [dags[i] for i in tr if labels[i] == 0]
    for method, weighting in GRAPHON_VARIANTS:
        est = GraphonEstimator(
            B=B, block_method=method, weighting=weighting, normalize=True
        ).fit(clean_tr_dags)
        scores = np.array([-est.log_likelihood(dags[i]) for i in te])
        results[f"graphon_{method}|{weighting}"] = bootstrap_auroc_ci(
            scores, y_te, seed=seed
        )

    # Supervised fair-ablation arm.
    clf = DagFeatureClassifier().fit(
        [trajs[i] for i in tr], [dags[i] for i in tr], labels[tr]
    )
    scores = np.array([clf.score(trajs[i], dags[i]) for i in te])
    results["dag_logreg"] = bootstrap_auroc_ci(scores, y_te, seed=seed)

    # Zero-shot baselines + structural statistics.
    for cls in ZERO_SHOT_BASELINES + STRUCTURAL_DETECTORS:
        det = cls()
        scores = np.array([det.score(trajs[i], dags[i]) for i in te])
        results[det.name] = bootstrap_auroc_ci(scores, y_te, seed=seed)

    results["_meta"] = (float(y_te.mean()), float(len(te)), float(labels.mean()))
    return results


def gate_decision(per_seed: list[dict]) -> str:
    baseline_names = [c().name for c in ZERO_SHOT_BASELINES]
    graphon_names = [f"graphon_{m}|{w}" for m, w in GRAPHON_VARIANTS]

    go_votes = 0
    conditional = False
    for res in per_seed:
        best_name = max(graphon_names, key=lambda k: res[k][0])
        g_pt, g_lo, _ = res[best_name]
        beats_all = all(
            g_pt >= res[b][0] + 0.05 and g_lo > res[b][2] for b in baseline_names
        )
        if g_pt >= 0.65 and beats_all:
            go_votes += 1
        lr_pt, lr_lo, lr_hi = res["dag_logreg"]
        if not (g_pt > lr_hi):  # graphon does not clearly beat logreg
            conditional = True
    if go_votes >= 2:
        return "CONDITIONAL-GO (graphon ~ dag_logreg)" if conditional else "GO"
    return "NO-GO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = []
    lines = [f"# e1_synth_gate — n={args.n}, seeds={SEEDS}, B={B}", ""]
    for seed in SEEDS:
        res = run_seed(seed, args.n)
        per_seed.append(res)
        te_rate, n_te, all_rate = res["_meta"]
        lines.append(
            f"## seed {seed}  (test n={int(n_te)}, hallucination rate "
            f"{te_rate:.2f} test / {all_rate:.2f} overall)"
        )
        for name, (pt, lo, hi) in sorted(
            ((k, v) for k, v in res.items() if k != "_meta"),
            key=lambda kv: -kv[1][0],
        ):
            lines.append(f"  {name:28s} AUROC {pt:.3f}  [{lo:.3f}, {hi:.3f}]")
        lines.append("")
        np.savez(
            OUT_DIR / f"seed{seed}.npz",
            **{k: np.array(v) for k, v in res.items() if k != "_meta"},
            seed=seed,
            n=args.n,
        )

    decision = gate_decision(per_seed)
    lines += [f"## GATE G1 DECISION: {decision}", ""]
    report = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
