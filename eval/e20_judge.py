"""e20: the LLM-judge baseline against V7 (exploratory).

    python3 -m eval.e20_judge

For each healthcare confirmatory corpus with judge scores
(results/llm_judge/<judge>_on_<corpus>/scores.json, eval/paid/run_llm_judge.py):
session AUROC on L_U and L of the judge's event probability (zero-shot,
scored once and reused across the split seeds' test sets) against V7,
intake_lr+fb and SelfCheckGPT on the same test rows; paired session-clustered
differences and the pooled stratified bootstrap; and per-commitment AUROC of
the judge's p(wrong) on confident commitments whose truth is user-dependent,
against V7's q_t. Writes results/e20_judge/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e11_confirmatory import (
    CONFIRM, MIN_SESSIONS, REPO, SampleRestricted, add_external, build_dag, load, mean_auc, pooled_rows, strat_boot,
)
from eval.e12_exploratory import IntakeLR
from eval.e7_common import SEEDS, TRAIN_FRAC, cluster_boot_diff, run_protocol
from eval.score_trajectories import split_task_ids
from pge.baselines.intake_posterior import IntakePosterior, v7
from pge.baselines.selfcheckgpt import SelfCheckGPT

JUDGE = "gpt-4o-mini"
OUT = REPO / "results" / "e20_judge"


def main():
    t0 = time.time()
    numbers, runs = {}, {"L_U": {}, "L": {}}
    lines = [f"# e20: LLM judge ({JUDGE}, zero-shot, reads intake + commitments + reactions) vs V7 (exploratory)", ""]
    for tag, path, seed, n in CONFIRM:
        jf = REPO / "results" / "llm_judge" / f"{JUDGE}_on_{path.stem}" / "scores.json"
        if not (path.exists() and jf.exists()):
            lines.append(f"> {tag}: corpus or judge scores missing; skipped.")
            continue
        J = json.loads(jf.read_text())
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        S = dom.USER_DEPENDENT_SCHEMA
        dags = [build_dag(t) for t in trajs]
        for lab, y in (("L_U", LU), ("L", L)):
            F = [("V7", lambda: v7(dom, lab)), ("intake_lr+fb", lambda: IntakeLR(True)),
                 ("SelfCheckGPT", lambda: SampleRestricted(SelfCheckGPT(), S) if lab == "L_U" else SelfCheckGPT())]
            R = run_protocol(trajs, y, F, dags=dags)
            assert add_external(R, "llm_judge", J["score_by_task"]), "judge scores do not cover the test sets"
            runs[lab][tag] = (R, len(trajs))
            yy, g = np.concatenate(R["y"]), np.concatenate(R["tid"])
            d = numbers.setdefault(lab, {}).setdefault(tag, {})
            d["auroc"] = {k: mean_auc(R, k) for k in R["auroc"]}
            d["diffs"] = {f"V7 - {b}": cluster_boot_diff(pooled_rows(R, "V7"), pooled_rows(R, b), yy, g, n=2000)
                          for b in ("llm_judge", "intake_lr+fb", "SelfCheckGPT")}
            d["diffs"]["llm_judge - SelfCheckGPT"] = cluster_boot_diff(pooled_rows(R, "llm_judge"),
                                                                        pooled_rows(R, "SelfCheckGPT"), yy, g, n=2000)
        # per commitment: judge p(wrong) vs V7 q_t on the same commitments
        ids = [t.task_id for t in trajs]
        a_j, a_v = [], []
        for sd in SEEDS:
            tr_ids, te_ids = split_task_ids(ids, TRAIN_FRAC, sd)
            tr = [t for t in trajs if t.task_id in tr_ids]
            det = v7(dom, "L_U").fit(tr, None, [LU[i] for i, t in enumerate(trajs) if t.task_id in tr_ids])
            pj, pv, w = [], [], []
            for t in trajs:
                if t.task_id not in te_ids:
                    continue
                s = sessions[t.task_id]
                U = dom.user_dependent_attrs(s.setup_answers)
                q = det.turn_q(t)
                pjt = J["p_wrong_by_task"][t.task_id]
                for k, turn in enumerate(t.turns):
                    if turn.probability >= 0.6 and turn.attribute in U:
                        pj.append(pjt[k]); pv.append(q[k]); w.append(int(turn.guess_value != s.user_state[turn.attribute]))
            a_j.append(auroc(np.array(pj), np.array(w)))
            a_v.append(auroc(np.array(pv), np.array(w)))
        numbers.setdefault("percommit", {})[tag] = {"judge": float(np.mean(a_j)), "V7": float(np.mean(a_v))}
        print(tag, {k: round(v, 3) for k, v in numbers["L_U"][tag]["auroc"].items()}, numbers["percommit"][tag],
              f"{time.time() - t0:.0f}s", flush=True)
    for lab in ("L_U", "L"):
        if not runs[lab]:
            continue
        dets = list(next(iter(numbers[lab].values()))["auroc"])
        lines += [f"## {lab}", "", "| corpus | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
        lines += [f"| {t} | " + " | ".join(f"{numbers[lab][t]['auroc'][k]:.3f}" for k in dets) + " |" for t in runs[lab]]
        lines.append("")
        for t in runs[lab]:
            lines.append(f"- {t}: " + "; ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
                                                for k, v in numbers[lab][t]["diffs"].items()))
        pool = [R for R, n in runs[lab].values() if n >= MIN_SESSIONS]
        if len(pool) >= 2:
            numbers[lab]["pooled"] = {}
            lines += ["", f"Pooled over {len(pool)} corpora:"]
            for a, b in (("V7", "llm_judge"), ("llm_judge", "SelfCheckGPT"), ("llm_judge", None)):
                ci = strat_boot(pool, a, b)
                q = f"AUROC({a})" if b is None else f"{a} - {b}"
                numbers[lab]["pooled"][q] = ci
                lines.append(f"- {q}: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        lines.append("")
    if numbers.get("percommit"):
        lines += ["## per commitment (confident, user-dependent; mean over split seeds)", ""]
        lines += [f"- {t}: judge {d['judge']:.3f}, V7 {d['V7']:.3f}" for t, d in numbers["percommit"].items()]
    lines.append(f"\nruntime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
