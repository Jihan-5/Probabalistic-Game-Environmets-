"""e13: develop V7 (intake-conditioned posterior) on the DEVELOPMENT corpora only.

    python3 -m eval.e13_v7_dev

Loads only eval.e11_confirmatory.DEV (seed 42, protocol v2) and refuses any
confirmatory path. Compares V6, V7 without the event restriction (intake cells
only), V7, ctx_lr+fb and the exploratory intake_lr+fb on L_U and L with the
e11 protocol (splits 42/123/456, paired session-clustered bootstrap), plus the
oracle V6-session for reference. Writes results/e13_v7_dev/summary.md.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.e11_confirmatory import DEV, REPO, V6Session, build_dag, load, mean_auc, pooled_rows
from eval.e12_exploratory import IntakeLR
from eval.e7_common import cluster_boot_diff, run_protocol
from pge.baselines.context_lr import ContextLR
from pge.baselines.intake_posterior import IntakePosterior
from pge.baselines.posterior import ChannelPosterior

OUT = REPO / "results" / "e13_v7_dev"


def main():
    t0 = time.time()
    lines = ["# e13: V7 development (DEVELOPMENT corpora only, seed 42, protocol v2)", ""]
    numbers = {}
    for tag, path, seed, n in DEV:
        assert "confirm" not in str(path), "e13 must not read confirmatory corpora"
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        V6Session.U = {t.task_id: dom.user_dependent_attrs(sessions[t.task_id].setup_answers) for t in trajs}
        dags = [build_dag(t) for t in trajs]
        S = dom.USER_DEPENDENT_SCHEMA
        for lab, y in (("L_U", LU), ("L", L)):
            ea = S if lab == "L_U" else None
            F = [("V6", lambda: ChannelPosterior(event_attrs=ea)),
                 ("V7-cells", lambda: IntakePosterior(dom, event_attrs=ea, delta_grid=(0.0,))),
                 ("V7", lambda: IntakePosterior(dom, event_attrs=ea)),
                 ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
                 ("intake_lr+fb", lambda: IntakeLR(True))]
            if lab == "L_U":
                F.append(("V6-session (oracle)", V6Session))
            R = run_protocol(trajs, y, F, dags=dags)
            res = {d: mean_auc(R, d) for d in R["auroc"]}
            yy, g = np.concatenate(R["y"]), np.concatenate(R["tid"])
            for a, b in (("V7", "V6"), ("V7", "ctx_lr+fb"), ("V7", "intake_lr+fb"), ("V7-cells", "V6")):
                ci = cluster_boot_diff(pooled_rows(R, a), pooled_rows(R, b), yy, g, n=2000)
                res[f"{a} - {b}"] = ci
            numbers.setdefault(lab, {})[tag] = res
            line = f"[{tag} {lab}] " + " ".join(f"{d} {v:.3f}" for d, v in res.items() if not isinstance(v, (list, tuple)))
            print(line, f"({time.time() - t0:.0f}s)", flush=True)
            lines += [f"## {tag}, {lab}", "", "| detector | AUROC |", "|---|---|"]
            lines += [f"| {d} | {v:.3f} |" for d, v in res.items() if not isinstance(v, (list, tuple))]
            lines += [""] + [f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in res.items()
                             if isinstance(v, (list, tuple))] + [""]
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
