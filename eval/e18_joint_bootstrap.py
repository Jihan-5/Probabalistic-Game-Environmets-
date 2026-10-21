"""e18: sensitivity of the pooled intervals to the shared sessions (exploratory).

    python3 -m eval.e18_joint_bootstrap

The healthcare confirmatory corpora share the same 600 seed-7 sessions (same
task ids, intakes and hidden facts; only the agent differs). The registered
pooled bootstrap (e11 strat_boot) resamples sessions independently within each
agent, which ignores that sharing. Here sessions are resampled JOINTLY across
agents (one draw of session ids applied to every agent), for every pooled
comparison in Table 3, on the same fitted detectors and test rows as e11/e16.
Also: a bootstrap two-sided p-value per comparison and Holm-adjusted p-values
over the family of pooled L_U difference tests (H2-H8 and the secondary
V6 - ctx_lr+fb). Writes results/e18_joint_bootstrap/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e11_confirmatory import (
    CONFIRM, REPO, SampleRestricted, build_dag, load, mean_auc, pooled_rows, rules_aware, strat_boot,
)
from eval.e12_exploratory import IntakeLR
from eval.e7_common import run_protocol
from pge.baselines.context_lr import ContextLR, PriorOnly
from pge.baselines.intake_posterior import IntakePosterior, v7
from pge.baselines.posterior import ChannelPosterior
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import SemanticEntropy

OUT = REPO / "results" / "e18_joint_bootstrap"
N_BOOT = 2000
LU_TESTS = [("H1", "SelfCheckGPT", None), ("H1", "semantic_entropy", None),
            ("H2", "V6", "V6-prior"), ("H2b", "ctx_lr+fb", "ctx_lr"), ("H3", "V6", "ctx_lr"),
            ("H4", "V6", "SelfCheckGPT"), ("H4", "V6", "semantic_entropy"),
            ("H4b", "ctx_lr+fb", "SelfCheckGPT"), ("H4b", "ctx_lr+fb", "semantic_entropy"),
            ("sec.", "V6", "ctx_lr+fb"),
            ("H5", "etaF (Bayes, feedback)", "eta0 (Bayes, no feedback)"),
            ("H6", "V6-session", "LR-session+fb"),
            ("H7", "V7", "V6"), ("H8", "V7", "intake_lr+fb"), ("H8'", "V7", "ctx_lr+fb")]
L_TESTS = [("H2", "V6", "V6-prior"), ("H3", "V6", "ctx_lr"), ("H4", "V6", "SelfCheckGPT"),
           ("H7", "V7", "V6"), ("H8", "V7", "intake_lr+fb")]


def factories(dom, lab):
    S = dom.USER_DEPENDENT_SCHEMA
    ea = S if lab == "L_U" else None

    def sampled(det):
        return SampleRestricted(det, S) if lab == "L_U" else det

    return [("V6", lambda: ChannelPosterior(event_attrs=ea)), ("V6-prior", lambda: PriorOnly(ChannelPosterior(event_attrs=ea))),
            ("ctx_lr", lambda: ContextLR()), ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
            ("SelfCheckGPT", lambda: sampled(SelfCheckGPT())), ("semantic_entropy", lambda: sampled(SemanticEntropy())),
            ("V7", lambda: v7(dom, lab)), ("intake_lr+fb", lambda: IntakeLR(True))]


def merge(R, R2):
    for k in ("auroc", "scores"):
        for d, v in R2[k].items():
            R[k][d] = v
    assert all((a == b).all() for a, b in zip(R["tid"], R2["tid"]))
    return R


def joint(Rs, a, b, n=N_BOOT, seed=0):
    tid = np.concatenate(Rs[0]["tid"])
    for R in Rs:
        assert (np.concatenate(R["tid"]) == tid).all()
    uniq, inv = np.unique(tid, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    prep = [(pooled_rows(R, a), pooled_rows(R, b) if b else None, np.concatenate(R["y"])) for R in Rs]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        per = [auroc(sa[idx], y[idx]) - (auroc(sb[idx], y[idx]) if sb is not None else 0.0) for sa, sb, y in prep]
        vals.append(np.mean(per))
    vals = np.array(vals)
    point = float(np.mean([mean_auc(R, a) - (mean_auc(R, b) if b else 0.0) for R in Rs]))
    p = float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))) if b else float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), p


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    t0 = time.time()
    runs = {"L_U": [], "L": []}
    for tag, path, seed, n in CONFIRM:
        if not path.exists():
            continue
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        dags = [build_dag(t) for t in trajs]
        R = run_protocol(trajs, LU, factories(dom, "L_U"), dags=dags)
        R = merge(R, rules_aware(trajs, sessions, LU, dags, float(trajs[0].meta.get("epsilon", 0.25))))
        runs["L_U"].append(R)
        runs["L"].append(run_protocol(trajs, L, factories(dom, "L"), dags=dags))
        print(tag, f"{time.time() - t0:.0f}s", flush=True)
    numbers, lines = {}, ["# e18: pooled intervals with sessions resampled jointly across agents (sensitivity)", ""]
    for lab, tests in (("L_U", LU_TESTS), ("L", L_TESTS)):
        Rs = runs[lab]
        rows = []
        for hid, a, b in tests:
            reg = strat_boot(Rs, a, b)
            jb = joint(Rs, a, b)
            rows.append((hid, a, b, reg, jb))
        diff_idx = [i for i, r in enumerate(rows) if r[2] is not None]
        adj = holm(np.array([rows[i][4][3] for i in diff_idx]))
        lines += [f"## {lab} (pooled over {len(Rs)} agents)", "",
                  "| id | comparison | registered bootstrap | joint-session bootstrap | p (joint) | Holm p |",
                  "|---|---|---|---|---|---|"]
        numbers[lab] = {}
        for k, (hid, a, b, reg, jb) in enumerate(rows):
            q = f"AUROC({a})" if b is None else f"{a} - {b}"
            hp = float(adj[diff_idx.index(k)]) if k in diff_idx else float("nan")
            numbers[lab][f"{hid}: {q}"] = {"registered": reg, "joint": jb[:3], "p_joint": jb[3], "p_holm": hp}
            lines.append(f"| {hid} | {q} | {reg[0]:+.3f} [{reg[1]:+.3f}, {reg[2]:+.3f}] | "
                         f"{jb[0]:+.3f} [{jb[1]:+.3f}, {jb[2]:+.3f}] | {jb[3]:.4f} | {hp:.4f} |")
        lines.append("")
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
