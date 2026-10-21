"""e13 sweep 2 (DEVELOPMENT corpora only, L_U): smaller membership thresholds, with
CIs against intake_lr+fb, and the label budget of the chosen threshold. Logged in
results/e13_v7_dev/sweep2.md whether kept or not."""

import numpy as np

from eval.e11_confirmatory import DEV, REPO, build_dag, load, mean_auc, pooled_rows
from eval.e12_exploratory import IntakeLR
from eval.e13_v7_dev_sweep import budget
from eval.e7_common import cluster_boot_diff, run_protocol
from pge.baselines.intake_posterior import IntakePosterior


def main():
    lines = ["# e13 sweep 2 (DEVELOPMENT corpora only, L_U)", ""]
    for tag, path, seed, n in DEV:
        assert "confirm" not in str(path)
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        S = dom.USER_DEPENDENT_SCHEMA
        F = [(f"V7m k{k}", (lambda k=k: IntakePosterior(dom, event_attrs=S, event="learned", min_cell=k)))
             for k in (2, 3, 5)] + [("intake_lr+fb", lambda: IntakeLR(True))]
        R = run_protocol(trajs, LU, F, dags=[build_dag(t) for t in trajs])
        y, g = np.concatenate(R["y"]), np.concatenate(R["tid"])
        lines += [f"## {tag}", ""]
        for d in R["auroc"]:
            lines.append(f"- {d}: {mean_auc(R, d):.3f}")
        for k in (2, 3, 5):
            ci = cluster_boot_diff(pooled_rows(R, f"V7m k{k}"), pooled_rows(R, "intake_lr+fb"), y, g, n=2000)
            lines.append(f"- V7m k{k} - intake_lr+fb: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        b = budget(trajs, LU, {"V7m k5": lambda: IntakePosterior(dom, event_attrs=S, event="learned", min_cell=5),
                               "intake_lr+fb": lambda: IntakeLR(True)})
        lines += ["", "label budget:"] + [f"- {k[0]} @ {k[1]}: {v:.3f}" for k, v in
                                          sorted(b.items(), key=lambda kv: (kv[0][1], kv[0][0]))] + [""]
        print("\n".join(lines[-14:]), flush=True)
    (REPO / "results" / "e13_v7_dev" / "sweep2.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
