"""e9_consistency_bound — per-commitment evidence for Theorem 1 ($0, analysis).

Theorem 1 (paper): any detector computed from the agent's context and its
own samples is bounded by what the context reveals; under systematic
fabrication (a near-degenerate predictive distribution) every
sampling-consistency score is near-constant, so its AUROC is near 1/2.

On underdetermined healthcare attributes (truth depends on hidden fields
absent from the agent's context), per-commitment truth is reconstructed
from the generator stream and we compare, for predicting whether THAT
commitment is wrong:
  * resample disagreement (k=3 cached resamples; what SelfCheckGPT and
    semantic entropy aggregate), and
  * one noisy feedback bit (eps = 0.25 flip channel).

Usage: python3 -m eval.e9_consistency_bound
Writes results/e9_consistency_bound/summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc, bootstrap_auroc_ci
from eval.e6_ceiling import load_with_truth
from pge.baselines.posterior import agreement_count, turn_signal
from pge.benchmarks.healthcare_v2 import UNDERDETERMINED

REPO = Path(__file__).resolve().parent.parent


def main():
    lines = ["# e9_consistency_bound — per-commitment AUROC on underdetermined attributes", ""]
    for name, n in (("70B_42", 600), ("8B_42", 200)):
        dis, wrong, fb = [], [], []
        for traj, w in load_with_truth(name, n, 42):
            for turn, wi in zip(traj.turns, w):
                c = agreement_count(turn)
                if c is None or turn.attribute not in UNDERDETERMINED:
                    continue
                dis.append(len(turn.samples) - c)
                wrong.append(wi)
                fb.append(-turn_signal(turn))
        dis, wrong, fb = (np.array(x, dtype=float) for x in (dis, wrong, fb))
        a_d, lo_d, hi_d = bootstrap_auroc_ci(dis, wrong.astype(int), n_resamples=2000, seed=0)
        a_f, lo_f, hi_f = bootstrap_auroc_ci(fb, wrong.astype(int), n_resamples=2000, seed=0)
        lines += [
            f"## healthcare {name}: {len(wrong)} commitments, {wrong.mean():.0%} wrong",
            f"  all 3 resamples agree: {np.mean(dis == 0):.1%} of commitments",
            f"  P(any disagreement | wrong) = {np.mean(dis[wrong == 1] > 0):.3f}   "
            f"P(any disagreement | right) = {np.mean(dis[wrong == 0] > 0):.3f}",
            f"  AUROC resample disagreement  {a_d:.3f} [{lo_d:.3f}, {hi_d:.3f}]",
            f"  AUROC one noisy feedback bit {a_f:.3f} [{lo_f:.3f}, {hi_f:.3f}]",
            "",
        ]
    out = REPO / "results" / "e9_consistency_bound" / "summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
