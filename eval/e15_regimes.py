"""e15: how detection depends on the user's behaviour (semi-synthetic).

    python3 -m eval.e15_regimes                 # development corpora
    python3 -m eval.e15_regimes --confirm       # confirmatory healthcare corpora

The agents' real trajectories are kept; only the reactions the DETECTORS see
are regenerated from the noise-free truth, with flip rate eps and reaction
rate rho (each commitment on a reactable attribute gets a reaction with
probability rho, independently of its truth). This is Theorem 2's setting:
the agent's trajectories are fixed. Sampling detectors read only resamples
and are unaffected. Reported on L_U: the exact Bayes posterior with
the regenerated reactions (oracle), V6, V7m, ctx_lr+fb, intake_lr+fb and
SelfCheckGPT, one regenerated draw per (eps, rho), splits 42/123/456.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import zlib

import numpy as np

from eval.e11_confirmatory import (
    CONFIRM, DEV, REPO, Fixed, SampleRestricted, bayes_LU, build_dag, load, mean_auc,
)
from eval.e12_exploratory import IntakeLR
from eval.e7_common import run_protocol
from pge.baselines.context_lr import ContextLR
from pge.baselines.intake_posterior import IntakePosterior
from pge.baselines.posterior import ChannelPosterior
from pge.baselines.selfcheckgpt import SelfCheckGPT

EPS = (0.05, 0.15, 0.25, 0.35, 0.45)
RHO = (1.0, 0.5, 0.25)


def regenerate(trajs, sessions, dom, eps, rho, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in trajs:
        t2 = copy.deepcopy(t)
        st = sessions[t.task_id].user_state
        for turn in t2.turns:
            if turn.attribute in dom.NO_FEEDBACK_ATTRS or rng.random() >= rho:
                s = 0
            else:
                s = 1 if turn.guess_value == st[turn.attribute] else -1
                if rng.random() < eps:
                    s = -s
            turn.signals = {"noisy_feedback": s}
        out.append(t2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)
    corpora = [c for c in CONFIRM if c[1].exists()] if args.confirm else DEV
    t0 = time.time()
    res = {}
    for tag, path, seed, n in corpora:
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        S = dom.USER_DEPENDENT_SCHEMA
        dags = [build_dag(t) for t in trajs]
        sc = None
        for rho in RHO:
            for eps in EPS:
                if rho < 1.0 and eps != 0.25:
                    continue  # rate sweep at the main noise level only
                tr2 = regenerate(trajs, sessions, dom, eps, rho, seed=zlib.crc32(f'{tag}|{eps}|{rho}'.encode()))
                eF = {t.task_id: bayes_LU(t, sessions[t.task_id].setup_answers, eps) for t in tr2}
                F = [("etaF (oracle)", lambda: Fixed("etaF", eF)),
                     ("V6", lambda: ChannelPosterior(event_attrs=S)),
                     ("V7m", lambda: IntakePosterior(dom, event_attrs=S, event="learned")),
                     ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
                     ("intake_lr+fb", lambda: IntakeLR(True))]
                if sc is None:
                    F.append(("SelfCheckGPT", lambda: SampleRestricted(SelfCheckGPT(), S)))
                R = run_protocol(tr2, LU, F, dags=dags)
                row = {d: mean_auc(R, d) for d in R["auroc"]}
                if sc is None:
                    sc = row["SelfCheckGPT"]
                row["SelfCheckGPT"] = sc
                res.setdefault(tag, {})[f"eps={eps},rho={rho}"] = row
                print(tag, eps, rho, {k: round(v, 3) for k, v in row.items()}, f"{time.time() - t0:.0f}s", flush=True)
    out = REPO / "results" / ("e15_regimes_confirm" if args.confirm else "e15_regimes_dev")
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"# e15 user-behaviour regimes ({'confirmatory' if args.confirm else 'development'} corpora; "
             "detector-side regenerated reactions, L_U AUROC)", ""]
    for tag, d in res.items():
        dets = list(next(iter(d.values())))
        lines += [f"## {tag}", "", "| regime | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
        lines += [f"| {k} | " + " | ".join(f"{v[x]:.3f}" for x in dets) + " |" for k, v in d.items()] + [""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
