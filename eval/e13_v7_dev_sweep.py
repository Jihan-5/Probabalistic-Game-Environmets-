"""e13 sweep: V7m variants and label budgets on the DEVELOPMENT corpora only (L_U).

    python3 -m eval.e13_v7_dev_sweep

Every variant tried is logged here (results/e13_v7_dev/sweep.md), whether kept or not.
"""

import numpy as np

from eval.e11_confirmatory import DEV, REPO, build_dag, load, mean_auc
from eval.e12_exploratory import IntakeLR
from eval.e7_common import SEEDS, run_protocol
from eval.score_trajectories import split_task_ids
from eval.bootstrap_ci import auroc
from pge.baselines.intake_posterior import IntakePosterior

VARIANTS = [("V7m k10 r2", dict(min_cell=10, rounds=2)), ("V7m k5 r2", dict(min_cell=5, rounds=2)),
            ("V7m k20 r2", dict(min_cell=20, rounds=2)), ("V7m k10 r3", dict(min_cell=10, rounds=3)),
            ("V7m k10 r2 halves", dict(min_cell=10, rounds=2, halves=True)),
            ("V7m k5 r2 halves", dict(min_cell=5, rounds=2, halves=True))]


def budget(trajs, y, makers, sizes=(25, 50, 100, 200), draws=5):
    ids = [t.task_id for t in trajs]
    out = {}
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids(ids, 0.7, seed)
        tr = np.array([i for i, t in enumerate(ids) if t in tr_ids])
        te = [i for i, t in enumerate(ids) if t in te_ids]
        for n in [m for m in sizes if m < len(tr)]:
            for d in range(draws):
                rng = np.random.default_rng((seed, n, d))
                for _ in range(50):
                    sub = rng.choice(tr, n, replace=False)
                    if 0 < y[sub].sum() < len(sub):
                        break
                else:
                    continue
                for name, mk in makers.items():
                    det = mk().fit([trajs[i] for i in sub], None, y[sub])
                    out.setdefault((name, n), []).append(auroc(np.array([det.score(trajs[i]) for i in te]), y[te]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    lines = ["# e13 sweep (DEVELOPMENT corpora only, L_U)", ""]
    for tag, path, seed, n in DEV:
        assert "confirm" not in str(path)
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        S = dom.USER_DEPENDENT_SCHEMA
        F = [(name, (lambda kw=kw: IntakePosterior(dom, event_attrs=S, event="learned", **kw))) for name, kw in VARIANTS]
        F.append(("intake_lr+fb", lambda: IntakeLR(True)))
        R = run_protocol(trajs, LU, F, dags=[build_dag(t) for t in trajs])
        res = {d: mean_auc(R, d) for d in R["auroc"]}
        print(tag, res, flush=True)
        lines += [f"## {tag}", ""] + [f"- {d}: {v:.3f}" for d, v in res.items()] + [""]
        mk = {"V7m k10 r2": lambda: IntakePosterior(dom, event_attrs=S, event="learned"),
              "V7m k10 r2 halves": lambda: IntakePosterior(dom, event_attrs=S, event="learned", halves=True),
              "intake_lr+fb": lambda: IntakeLR(True)}
        b = budget(trajs, LU, mk)
        print(tag, b, flush=True)
        lines += [f"label budget ({tag}):", ""] + [f"- {k[0]} @ {k[1]}: {v:.3f}" for k, v in sorted(b.items(), key=lambda kv: (kv[0][1], kv[0][0]))] + [""]
    (REPO / "results" / "e13_v7_dev" / "sweep.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
