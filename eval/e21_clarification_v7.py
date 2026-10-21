"""e21: Amendment 7, clarification with the V7 picker, analysed against the registered run.

    python3 -m eval.e21_clarification_v7

The V7-picker run (eval/paid/run_clarification_closed_loop.py --picker v7
--conditions posterior@1 posterior@2, out results/e8_closed_loop/70B_7_v7picker)
reruns only the conditions whose picks depend on the picker. Every other
condition (cached, random@1, lowconf@1, no_question@b, detailed_questions,
rerun_full) does not depend on the picker and is taken from the registered run
(results/e8_closed_loop/70B_7): same 120 test sessions, same patient noise
streams. Paired bootstrap over sessions (2000), as in the registered summary.
Primary outcome: wrong confident user-dependent commitments per session.
Writes results/e21_clarification_v7/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json

import numpy as np

from eval.e11_confirmatory import REPO
from eval.paid.run_clarification_closed_loop import paired_boot
from pge.trajectory import load_jsonl

REG = REPO / "results" / "e8_closed_loop" / "70B_7"
NEW = REPO / "results" / "e8_closed_loop" / "70B_7_v7picker"
OUT = REPO / "results" / "e21_clarification_v7"


def load(d, cond):
    f = d / f"{cond.replace('@', '_at')}.jsonl"
    return {t.task_id: t for t in load_jsonl(f)} if f.exists() else None


def main():
    conds = {f"V7 {c}": load(NEW, c) for c in ("posterior@1", "posterior@2")}
    conds.update({c: load(REG, c) for c in ("cached", "posterior@1", "posterior@2", "random@1", "lowconf@1",
                                          "no_question@1", "no_question@2", "detailed_questions", "rerun_full")})
    missing = [c for c, v in conds.items() if v is None]
    if missing:
        raise SystemExit(f"missing conditions: {missing}")
    tids = sorted(set.intersection(*[set(v) for v in conds.values()]), key=lambda s: int(s.split("_")[1]))
    out = {"n_sessions": len(tids), "conditions": {}, "comparisons": {}}
    lines = ["# e21: clarification with the V7 picker (Amendment 7) vs the registered run", "",
             f"paired sessions: {len(tids)}", "",
             "| condition | questions/sess | L_U rate | wrong conf. user-dep./sess (primary) | L rate |",
             "|---|---|---|---|---|"]
    col = {}
    for c, d in conds.items():
        rows = [d[t] for t in tids]
        lu = np.array([r.meta["label_user"] for r in rows], float)
        wcu = np.array([r.meta["wrong_confident_user"] for r in rows], float)
        lab = np.array([r.eval_label for r in rows], float)
        col[c] = (lu, wcu, lab)
        nq = float(np.mean([r.meta["n_questions"] for r in rows]))
        out["conditions"][c] = {"questions": nq, "L_U": float(lu.mean()), "wcu": float(wcu.mean()), "L": float(lab.mean())}
        lines.append(f"| {c} | {nq:.2f} | {lu.mean():.3f} | {wcu.mean():.3f} | {lab.mean():.3f} |")
    comps = [("H9", "V7 posterior@1", "random@1"), ("--", "V7 posterior@1", "no_question@1"),
             ("--", "V7 posterior@1", "lowconf@1"), ("--", "V7 posterior@1", "posterior@1"),
             ("--", "V7 posterior@2", "no_question@2"), ("--", "V7 posterior@2", "detailed_questions"),
             ("--", "V7 posterior@2", "posterior@2"), ("--", "V7 posterior@1", "cached")]
    lines += ["", "Paired bootstrap over sessions (2000), A - B, negative = A better:", "",
              "| id | A | B | d wrong conf. user-dep./sess | d L_U rate | d L rate |", "|---|---|---|---|---|---|"]
    for hid, a, b in comps:
        r = [paired_boot(col[a][k], col[b][k]) for k in (1, 0, 2)]
        out["comparisons"][f"{hid}: {a} - {b}"] = {"wcu": r[0], "L_U": r[1], "L": r[2]}
        lines.append(f"| {hid} | {a} | {b} | " + " | ".join(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, lo, hi in r) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
