"""e6_ceiling — how much signal the healthcare feedback channel can carry ($0).

ANALYSIS ONLY. Reconstructs each cached session's noise-free per-turn truth
from the generator seed stream (same as eval/generate_healthcare_v2.py) and
computes the AUROC of the exact label-event posterior with ORACLE priors
(estimated in-sample from true per-turn errors) and the true flip rate
eps=0.25. This is an optimistic ceiling for any detector that sees only
(confidence, committed value, feedback). Truth never feeds a detector.

Usage: python3 -m eval.e6_ceiling
Writes results/e6_posterior/ceiling.txt
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from pge.baselines.posterior import (
    HIGH,
    conf_bucket,
    label_threshold,
    poisson_binomial_tail,
    turn_signal,
)
from pge.benchmarks.healthcare_v2 import sample_session_v2
from pge.trajectory import load_jsonl

REPO = Path(__file__).resolve().parent.parent
CORPORA = [("70B_42", 600, 42), ("8B_42", 200, 42)]


def load_with_truth(name, n, seed):
    by_id = {t.task_id: t for t in load_jsonl(REPO / "data/trajectories/healthcare_v2" / f"{name}.jsonl")}
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = sample_session_v2(rng, i)
        t = by_id[f"hcv2_{s.profile_id}"]
        assert [tt.attribute for tt in t.turns] == s.schedule, "session stream mismatch"
        out.append((t, [int(tt.guess_value != s.user_state[tt.attribute]) for tt in t.turns]))
    return out


def oracle_auroc(D, keyfn, eps=0.25):
    num, den = defaultdict(float), defaultdict(float)
    for t, w in D:
        for tt, wi in zip(t.turns, w):
            num[keyfn(tt)] += wi
            den[keyfn(tt)] += 1
    pi = {k: float(np.clip(num[k] / den[k], 0.01, 0.99)) for k in den}
    scores = []
    for t, _ in D:
        q = []
        for tt in t.turns:
            if tt.probability < HIGH:
                continue
            p0, s = pi[keyfn(tt)], turn_signal(tt)
            l1 = 1.0 if s == 0 else ((1 - eps) if s < 0 else eps)
            l0 = 1.0 if s == 0 else (eps if s < 0 else (1 - eps))
            q.append(p0 * l1 / (p0 * l1 + (1 - p0) * l0))
        scores.append(poisson_binomial_tail(np.array(q), label_threshold(len(q))) if q else 0.0)
    return auroc(np.array(scores), np.array([t.eval_label for t, _ in D]))


def main():
    lines = ["# e6_ceiling — oracle-prior label-event posterior (in-sample, optimistic)", ""]
    for name, n, seed in CORPORA:
        D = load_with_truth(name, n, seed)
        flips = obs = 0
        for t, w in D:
            for tt, wi in zip(t.turns, w):
                s = turn_signal(tt)
                if s:
                    obs += 1
                    flips += (s < 0) != (wi == 1)
        lines.append(f"## healthcare {name}: empirical flip rate {flips / obs:.3f} ({obs} observed turns)")
        lines.append(f"  oracle priors by (attribute, bucket)         AUROC "
                     f"{oracle_auroc(D, lambda tt: (tt.attribute, conf_bucket(tt.probability))):.3f}")
        lines.append(f"  oracle priors by (attribute, value, bucket)  AUROC "
                     f"{oracle_auroc(D, lambda tt: (tt.attribute, tt.guess_value, conf_bucket(tt.probability))):.3f}")
        lines.append("")
    out = REPO / "results" / "e6_posterior" / "ceiling.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
