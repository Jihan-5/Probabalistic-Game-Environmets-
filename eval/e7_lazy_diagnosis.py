"""e7b follow-up — why V3 collapses under the one-sided "lazy user" channel ($0).

Diagnosis only (no new detector). Same regenerated channel as e7b
(setting '<tag>|lazy user P(-|w)=0.6, no +1', draw 0), split seed 123.

Usage: python3 -m eval.e7_lazy_diagnosis
Writes results/e7_channel_misspec/lazy_diagnosis.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.e7_common import auroc, channel_rng, load_corpus, regenerate_channel, with_signals
from eval.score_trajectories import split_task_ids
from pge.baselines.posterior import IndependentPosterior, turn_signal

REPO = Path(__file__).resolve().parent.parent


def main():
    lines = ["# e7b lazy-user channel: V3 failure diagnosis (draw 0, split seed 123)", ""]
    for tag in ("70B", "8B"):
        trajs, truth = load_corpus(tag)
        y = np.array([t.eval_label for t in trajs])
        sig, _ = regenerate_channel("lazy", 0.6, trajs, truth, channel_rng(f"{tag}|lazy user P(-|w)=0.6, no +1", 0))
        D = [with_signals(t, s) for t, s in zip(trajs, sig)]
        tr_ids, te_ids = split_task_ids([t.task_id for t in D], 0.7, 123)
        tr = [i for i, t in enumerate(D) if t.task_id in tr_ids]
        te = [i for i, t in enumerate(D) if t.task_id in te_ids]
        det = IndependentPosterior(by_value=True).fit([D[i] for i in tr], None, y[tr])
        pis = np.array(list(det.pi.values()))
        s = np.array([det.score(D[i]) for i in te])
        nh = np.array([sum(tt.probability >= 0.6 for tt in D[i].turns) for i in te])
        n_obs_hi = np.array([sum(tt.probability >= 0.6 and turn_signal(tt) != 0 for tt in D[i].turns) for i in te])
        frac_obs_hi = np.array([n_obs_hi[k] / max(nh[k], 1) for k in range(len(te))])
        lines += [
            f"## {tag}",
            f"  learned eps {det.eps} (grid max 0.45); bucket priors {det.pi_bucket}",
            f"  value/attr priors: min {pis.min():.3f} median {np.median(pis):.3f} max {pis.max():.3f}",
            f"  test scores quantiles (0/10/50/90/100%): {np.round(np.quantile(s, [0, .1, .5, .9, 1]), 4).tolist()}",
            f"  AUROC V3 score {auroc(s, y[te]):.3f}; AUROC(-#high-conf turns) {auroc(-nh, y[te]):.3f}; "
            f"corr(score, #high-conf) {np.corrcoef(s, nh)[0, 1]:+.2f}",
            f"  AUROC(fraction of high-conf turns that received a -1) {auroc(frac_obs_hi, y[te]):.3f} "
            "(the informative-missingness statistic V3 cannot use)",
            "",
        ]
    lines += [
        "Mechanism: with +1 never emitted, every observed signal is -1, so the method-of-moments prior",
        "(n_neg/n_obs - eps)/(1 - 2 eps) is >= 1 and clips to 0.99 for every key; eps runs to the grid edge.",
        "Every high-confidence turn then has posterior error ~1 whether or not it was contradicted, the",
        "Poisson-binomial tail saturates at ~1, and the residual ranking is round-off that tracks the number",
        "of high-confidence turns (on 70B that count is anti-correlated with the label, hence AUROC < 0.5; on 8B",
        "it is roughly uninformative). V3's channel model treats 0 (unobserved) as ignorable; under this channel",
        "silence is informative (missing not at random). A simple statistic that uses it (fraction of high-conf",
        "turns contradicted) reaches 0.80/0.82, and supervised dag_logreg (features include the negative-score",
        "fraction and an unobserved-mass feature) reaches 0.83/0.80 in e7b. Not fixed here: a one-sided channel",
        "model (P(-1|wrong)=c, P(-1|right)=0, P(0|.) informative) would be a new detector, not an ablation.",
    ]
    out = REPO / "results" / "e7_channel_misspec" / "lazy_diagnosis.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
