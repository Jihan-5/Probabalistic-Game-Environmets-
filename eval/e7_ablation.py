"""e7d — clean protocol ablation of the value-conditioned posterior (V3), $0.

Healthcare 70B and 8B, original labels and recorded feedback, e6 protocol
(seeds 42/123/456), paired bootstrap of AUROC(variant) - AUROC(full V3)
with test sets pooled over seeds (item-level = e6 protocol; task-level =
resample unique task_ids). Every variant is a subclass/config of
IndependentPosterior defined here; posterior.py is untouched.

Variants (all fixed a priori):
  full V3                 IndependentPosterior(by_value=True), k=5
  - value (V1)            prior by (attribute, bucket) only
  flat prior              one global error rate (MoM over ALL observed turns),
                          no attribute / value / bucket
  - event targeting       score = mean posterior error over p >= 0.6 turns
                          instead of the Poisson-binomial tail.
                          [a] eps selected by the ablated score's own train BCE
                          [b] priors + eps from the full-V3 fit, only the
                              final aggregation swapped (isolates the score)
  - high-conf restriction event computed over ALL turns
                          (>= 50% of all commitments wrong), label unchanged
                          [a] own-BCE eps   [b] full-V3 fit, score swapped
  eps fixed 0.25          no label use at all (fully unsupervised detector)
  eps fixed 0.10 / 0.40   misspecified channel noise, no label use
  + samples (V4)          IndependentPosterior(by_value=True, use_samples=True)
  shrinkage k=1 / k=20

Usage: python3 -m eval.e7_ablation [70B] [8B]
Writes results/e7_ablation/summary.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from eval.e7_common import diff_cis, fmt_ci, load_corpus, masked, mean_auprc, mean_auroc, run_protocol
from pge.baselines.posterior import (
    BUCKETS, IndependentPosterior, emission, label_threshold,
    poisson_binomial_tail, turn_signal,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7_ablation"


class Ablated(IndependentPosterior):
    """V3 with one component switched off.

    agg        'tail' (event probability) | 'mean' (mean posterior error)
    event_all  event over all turns instead of p >= high
    flat       single global prior
    eps_fixed  skip label-based eps selection; use this eps
    v3_fit     fit exactly as full V3 (tail, high-conf), apply the ablation
               only at score time
    """

    def __init__(self, name, agg="tail", event_all=False, flat=False,
                 eps_fixed=None, v3_fit=False, **kw):
        kw.setdefault("by_value", not flat)
        super().__init__(**kw)
        self.name = name
        self.agg, self.event_all, self.flat = agg, event_all, flat
        self.eps_fixed, self.v3_fit = eps_fixed, v3_fit
        self._ablate = True

    def _fit_pi(self, trajs, eps):
        if not self.flat:
            return super()._fit_pi(trajs, eps)
        n = o = 0.0
        for t in trajs:
            for turn in t.turns:
                s = turn_signal(turn)
                if s:
                    n += s < 0
                    o += 1
        g = float(np.clip((n / o - eps) / max(1 - 2 * eps, 1e-3), 0.01, 0.99)) if o else 0.5
        return {}, {b: g for b in range(len(BUCKETS) + 1)}

    def _score(self, traj):
        if not self._ablate:
            return super()._score(traj)
        q = []
        for turn in traj.turns:
            if not self.event_all and turn.probability < self.high:
                continue
            pi = self._prior(turn)
            s = turn_signal(turn)
            l1, l0 = emission(s, 1, self.eps), emission(s, 0, self.eps)
            q.append(pi * l1 / (pi * l1 + (1 - pi) * l0))
        if not q:
            return 0.0
        q = np.array(q)
        if self.agg == "mean":
            return float(q.mean())
        return poisson_binomial_tail(q, label_threshold(len(q), self.frac))

    def fit(self, trajectories, dags=None, labels=None):
        if self.eps_fixed is not None:  # no labels used
            self.eps = self.eps_fixed
            self.pi, self.pi_bucket = self._fit_pi(trajectories, self.eps)
            return self
        self._ablate = not self.v3_fit
        super().fit(trajectories, dags, labels)
        self._ablate = True
        return self


VARIANTS = [
    ("full V3", lambda: IndependentPosterior(by_value=True)),
    ("- value conditioning (V1)", lambda: IndependentPosterior()),
    ("flat prior (global rate)", lambda: Ablated("flat", flat=True)),
    ("- event targeting [a] mean post., own eps", lambda: Ablated("mean_a", agg="mean")),
    ("- event targeting [b] mean post., V3 fit", lambda: Ablated("mean_b", agg="mean", v3_fit=True)),
    ("- high-conf restriction [a] own eps", lambda: Ablated("all_a", event_all=True)),
    ("- high-conf restriction [b] V3 fit", lambda: Ablated("all_b", event_all=True, v3_fit=True)),
    ("eps fixed 0.25 (no labels used)", lambda: Ablated("eps25", eps_fixed=0.25)),
    ("eps fixed 0.10 (misspecified, no labels)", lambda: Ablated("eps10", eps_fixed=0.10)),
    ("eps fixed 0.40 (misspecified, no labels)", lambda: Ablated("eps40", eps_fixed=0.40)),
    ("+ samples (V4)", lambda: IndependentPosterior(by_value=True, use_samples=True)),
    ("shrinkage k=1", lambda: IndependentPosterior(prior_strength=1.0, by_value=True)),
    ("shrinkage k=20", lambda: IndependentPosterior(prior_strength=20.0, by_value=True)),
]


def run_corpus(tag, lines):
    trajs, _ = load_corpus(tag)  # truth unused here
    labels = np.array([t.eval_label for t in trajs])
    det_trajs = [masked(t) for t in trajs]
    t0 = time.time()
    R = run_protocol(det_trajs, labels, VARIANTS)
    lines += [f"## healthcare {tag} (n={len(trajs)}, hall rate {labels.mean():.3f}; {time.time() - t0:.0f}s)", "",
              "| variant | AUROC (mean) | per seed 42/123/456 | AUPRC | vs full V3 (item CI) | vs full V3 (task CI) | eps per seed |",
              "|---|---|---|---|---|---|---|"]
    for name, _ in VARIANTS:
        per = "/".join(f"{x:.3f}" for x in R["auroc"][name])
        if name == "full V3":
            d = ("—", "—")
        else:
            item, clus = diff_cis(R, name, "full V3")
            d = (fmt_ci(item), fmt_ci(clus))
        eps = R["eps"].get(name, [])
        lines.append(f"| {name} | {mean_auroc(R, name):.3f} | {per} | {mean_auprc(R, name):.3f} | {d[0]} | {d[1]} | "
                     f"{'/'.join(f'{e:.3f}' for e in eps)} |")
    lines.append("")
    print(f"[{tag}] done ({time.time() - t0:.0f}s)", flush=True)


def main():
    tags = [a for a in sys.argv[1:] if a in ("70B", "8B")] or ["70B", "8B"]
    t0 = time.time()
    lines = [
        "# e7d protocol ablation of V3 (offline, $0)",
        "",
        "Original eval_label, recorded feedback. e6 protocol (split seeds 42/123/456, fit train, test once).",
        "Difference = AUROC(variant) - AUROC(full V3); paired bootstrap 2000, test sets pooled over seeds;",
        "item = rows resampled (e6), task = unique task_ids resampled. * = CI excludes 0.",
        "eps grid for learned-eps variants: 0.05..0.45 step 0.025 (posterior.EPS_GRID). Nothing tuned on test.",
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
