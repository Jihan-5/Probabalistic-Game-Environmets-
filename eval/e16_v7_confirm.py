"""e16: Amendment 6 on the confirmatory corpora (V7 frozen at commit d32ea33).

    python3 -m eval.e16_v7_confirm

Runs once per available confirmatory corpus (healthcare and travel), with
e11's splits and bootstrap. L_U: V6, V7, ctx_lr, ctx_lr+fb, intake_lr+fb,
SelfCheckGPT, semantic entropy. L: the same with V7's L form. H7 = V7 - V6,
H8 = V7 - intake_lr+fb, H8' = V7 - ctx_lr+fb, pooled within domain. Also,
without claims: V7 vs the sampling detectors, the label-budget curve on e11's
subsets, and per-commitment AUROC/ECE of V7's q_t. Writes
results/e16_v7_confirm/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e11_confirmatory import (
    CONFIRM, MIN_SESSIONS, REPO, TRAVEL, SampleRestricted, build_dag, load, mean_auc, pooled_rows, strat_boot,
)
from eval.e12_exploratory import IntakeLR, ece
from eval.e13_v7_dev_sweep import budget
from eval.e7_common import SEEDS, TRAIN_FRAC, cluster_boot_diff, run_protocol
from eval.score_trajectories import split_task_ids
from pge.baselines.context_lr import ContextLR
from pge.baselines.intake_posterior import v7
from pge.baselines.posterior import ChannelPosterior
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import SemanticEntropy

OUT = REPO / "results" / "e16_v7_confirm"
COMPS = [("H7", "V7", "V6"), ("H8", "V7", "intake_lr+fb"), ("H8'", "V7", "ctx_lr+fb"),
         ("--", "V7", "SelfCheckGPT"), ("--", "V7", "semantic_entropy")]


def factories(dom, lab):
    S = dom.USER_DEPENDENT_SCHEMA
    ea = S if lab == "L_U" else None

    def sampled(det):
        return SampleRestricted(det, S) if lab == "L_U" else det

    return [("V6", lambda: ChannelPosterior(event_attrs=ea)), ("V7", lambda: v7(dom, lab)),
            ("ctx_lr", lambda: ContextLR()), ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
            ("intake_lr+fb", lambda: IntakeLR(True)),
            ("SelfCheckGPT", lambda: sampled(SelfCheckGPT())), ("semantic_entropy", lambda: sampled(SemanticEntropy()))]


def per_commitment_v7(trajs, LU, sessions, dom):
    ids = [t.task_id for t in trajs]
    qs, ws = [], []
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids(ids, TRAIN_FRAC, seed)
        tr = [t for t in trajs if t.task_id in tr_ids]
        det = v7(dom, "L_U").fit(tr, None, [LU[i] for i, t in enumerate(trajs) if t.task_id in tr_ids])
        for t in trajs:
            if t.task_id not in te_ids:
                continue
            s = sessions[t.task_id]
            U = dom.user_dependent_attrs(s.setup_answers)
            for turn, q in zip(t.turns, det.turn_q(t)):
                if turn.probability >= det.high and turn.attribute in U:
                    qs.append(float(q))
                    ws.append(int(turn.guess_value != s.user_state[turn.attribute]))
    qs, ws = np.array(qs), np.array(ws)
    return {"auroc_q": float(auroc(qs, ws)), "ece_q": ece(qs, ws), "n": int(len(ws)) // len(SEEDS)}


def block(corpora, name, lines, numbers):
    runs = {"L_U": {}, "L": {}}
    for tag, path, seed, n in corpora:
        if not path.exists():
            lines.append(f"> {tag}: not found; skipped.")
            continue
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        from pge.baselines.intake_posterior import IntakePosterior
        IntakePosterior.INTAKE = IntakeLR.SETUP
        dags = [build_dag(t) for t in trajs]
        for lab, y in (("L_U", LU), ("L", L)):
            R = run_protocol(trajs, y, factories(dom, lab), dags=dags)
            runs[lab][tag] = (R, len(trajs))
            d = numbers.setdefault(name, {}).setdefault(lab, {}).setdefault(tag, {})
            d["auroc"] = {k: mean_auc(R, k) for k in R["auroc"]}
            yy, g = np.concatenate(R["y"]), np.concatenate(R["tid"])
            d["diffs"] = {f"{a} - {b}": cluster_boot_diff(pooled_rows(R, a), pooled_rows(R, b), yy, g, n=2000)
                          for _, a, b in COMPS}
            print(f"[{name} {tag} {lab}] " + " ".join(f"{k} {v:.3f}" for k, v in d["auroc"].items()), flush=True)
        if len(trajs) >= MIN_SESSIONS:
            mk = {"V7": lambda: v7(dom, "L_U"), "intake_lr+fb": lambda: IntakeLR(True),
                  "ctx_lr+fb": lambda: ContextLR(with_feedback=True)}
            b = budget(trajs, LU, mk)
            numbers[name].setdefault("budget", {})[tag] = {f"{k[0]}@{k[1]}": v for k, v in b.items()}
        numbers[name].setdefault("percommit_v7", {})[tag] = per_commitment_v7(trajs, LU, sessions, dom)
    for lab in ("L_U", "L"):
        if not runs[lab]:
            continue
        lines += [f"## {name}: {lab}", "", "| corpus | " + " | ".join(k for k in numbers[name][lab][next(iter(runs[lab]))]["auroc"]) + " |",
                  "|---|" + "---|" * len(numbers[name][lab][next(iter(runs[lab]))]["auroc"])]
        for tag in runs[lab]:
            a = numbers[name][lab][tag]["auroc"]
            lines.append(f"| {tag} | " + " | ".join(f"{v:.3f}" for v in a.values()) + " |")
        lines.append("")
        for tag in runs[lab]:
            lines.append(f"- {tag}: " + "; ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
                                                  for k, v in numbers[name][lab][tag]["diffs"].items()))
        pool = [R for R, n in runs[lab].values() if n >= MIN_SESSIONS]
        if len(pool) >= 2:
            lines += ["", f"Pooled over {len(pool)} corpora (stratified cluster bootstrap):", ""]
            for hid, a, b in COMPS + [("--", "V7", None), ("--", "intake_lr+fb", None)]:
                ci = strat_boot(pool, a, b)
                q = f"AUROC({a})" if b is None else f"{a} - {b}"
                numbers[name][lab].setdefault("pooled", {})[f"{hid}: {q}"] = ci
                lines.append(f"- {hid} {q}: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        lines.append("")
    if numbers.get(name, {}).get("budget"):
        lines += [f"## {name}: L_U label budget (descriptive)", ""]
        for tag, d in numbers[name]["budget"].items():
            lines.append(f"- {tag}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(d.items())))
        lines.append("")
    if numbers.get(name, {}).get("percommit_v7"):
        lines += [f"## {name}: per-commitment q_t of V7", ""] + [
            f"- {tag}: AUROC {d['auroc_q']:.3f}, ECE {d['ece_q']:.3f} (n per split {d['n']})"
            for tag, d in numbers[name]["percommit_v7"].items()] + [""]


def main():
    t0 = time.time()
    lines = ["# e16: Amendment 6 (V7, frozen at d32ea33) on the confirmatory corpora", ""]
    numbers = {}
    block(CONFIRM, "healthcare", lines, numbers)
    block(TRAVEL, "travel", lines, numbers)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
