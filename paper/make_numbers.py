"""Write numbers.tex: \\setval{key}{value} for every number the paper cites.

Sources (repo): results/e11_confirmatory/numbers.json (confirmatory, prefix C)
and dev_numbers.json (development, prefix D). A key the paper uses but this
file does not define prints as a red ?? in the PDF, so no number can be typed
by hand.

Usage: python3 make_numbers.py /path/to/repo
"""

import json
import math
import sys
from pathlib import Path

TAG = {"70B": "70B", "8B": "8B", "qwen72b": "qwen", "gpt-4o-mini": "gpt", "mistral-small-24b": "mistral",
       "70B (dev)": "70B", "8B (dev)": "8B",
       "70B LLM-patient (profile)": "pat", "70B LLM-patient (informed)": "patinf",
       "gpt-4o-mini (travel)": "gpt", "8B (travel)": "8B",
       "70B LLM-patient v1 pilot (profile)": "patv1", "70B LLM-patient v1 pilot (informed)": "patv1inf",
       "70B LLM-patient v2 (profile)": "patv2", "70B LLM-patient v2 pilot (profile)": "patv2pilot"}


def a3(x, lead=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = f"{x:.3f}"
    return s if lead else s.replace("0.", ".", 1)


def sd(x):
    return f"{x:+.3f}".replace("+0.", "+.").replace("-0.", "$-$.")


def ci(t):
    m, lo, hi = t
    star = "^{*}" if (lo > 0 or hi < 0) else ""
    return f"{sd(m)}\\,[{sd(lo)}, {sd(hi)}]" + (f"$^{{*}}$" if star else "")


def main(repo):
    out = []

    def put(k, v):
        if v is not None:
            out.append(f"\\setval{{{k}}}{{{v}}}")

    blocks = []
    for pre, fn in (("C", "numbers.json"), ("D", "dev_numbers.json")):
        f = Path(repo) / "results" / "e11_confirmatory" / fn
        if f.exists():
            N = json.loads(f.read_text())
            blocks.append((pre, N))
            if "travel_v1" in N:
                blocks.append(("T", N["travel_v1"]))  # second domain: same structure
            if "llm_patient" in N:
                blocks.append(("P", N["llm_patient"]))  # LLM-simulated patient corpora
    for pre, N in blocks:
        for tag, d in N.get("corpora", {}).items():
            t = TAG.get(tag, tag)
            put(f"{pre}/N/{t}/n", d["n"])
            put(f"{pre}/N/{t}/L", f"{100 * d['L']:.1f}")
            put(f"{pre}/N/{t}/LU", f"{100 * d['L_U']:.1f}")
        for lab in ("L_U", "L"):
            for tag, d in N.get(lab, {}).items():
                if tag == "pooled":
                    for q, c in d.items():
                        put(f"{pre}/{lab}/pooled/{q}", ci(c) if " - " in q else
                            f"{a3(c[0])}\\,[{a3(c[1])}, {a3(c[2])}]")
                    continue
                t = TAG.get(tag, tag)
                for det, v in d.get("auroc", {}).items():
                    put(f"{pre}/{lab}/{t}/{det}", a3(v))
                    put(f"t:{pre}/{lab}/{t}/{det}", a3(v, lead=False))
                for q, c in d.get("diffs", {}).items():
                    put(f"{pre}/{lab}/{t}/{q}", ci(c))
        for tag, d in N.get("rules_aware", {}).items():
            if tag == "pooled":
                for q, c in d.items():
                    put(f"{pre}/RA/pooled/{q}", ci(c))
                continue
            t = TAG.get(tag, tag)
            for k, v in d.items():
                put(f"{pre}/RA/{t}/{k}", ci(v) if isinstance(v, list) else a3(v))
                if not isinstance(v, list):
                    put(f"t:{pre}/RA/{t}/{k}", a3(v, lead=False))
        for tag, d in N.get("percommit", {}).items():
            t = TAG.get(tag, tag)
            put(f"{pre}/PC/{t}/n", d["n"])
            for k in ("wrong", "all_agree"):
                put(f"{pre}/PC/{t}/{k}", f"{100 * d[k]:.1f}")
            for k in ("e1", "e0", "cor1_bound", "auroc_disagreement", "auroc_feedback_bit"):
                put(f"{pre}/PC/{t}/{k}", a3(d[k]))
        for lab, dd in N.get("learning_curve", {}).items():
            for tag, d in dd.items():
                t = TAG.get(tag, tag)
                for k, v in d.items():
                    put(f"{pre}/LC/{lab}/{t}/{k}", a3(v))
        for lab, d in N.get("k10", {}).items():
            for k, v in d.items():
                put(f"{pre}/K10/{lab}/{k}", a3(v))
        # ---- derived values over the agents of this block (ranges, means), so text never types them
        def rng(vals, fmt=lambda x: f"{x:.3f}"):
            vals = [v for v in vals if v is not None]
            return f"{fmt(min(vals))}--{fmt(max(vals))}" if vals else None
        pc = N.get("percommit", {})
        for k in ("auroc_disagreement", "auroc_feedback_bit", "e1", "e0", "cor1_bound"):
            put(f"{pre}/PC/range/{k}", rng([d.get(k) for d in pc.values()]))
        put(f"{pre}/PC/range/all_agree", rng([d.get("all_agree") for d in pc.values()], lambda x: f"{100 * x:.0f}"))
        ra, lu = N.get("rules_aware", {}), N.get("L_U", {})
        agents = [t for t in ra if t != "pooled" and t in lu]
        put(f"{pre}/derived/V6session_gap/range",
            rng([ra[t].get("V6-session", 0) - lu[t]["auroc"].get("V6", 0) for t in agents]))
        for k in ("eta0 (Bayes, no feedback)", "etaF (Bayes, feedback)"):
            put(f"{pre}/RA/range/{k}", rng([ra[t].get(k) for t in agents]))
        put(f"{pre}/RA/range/etaF - eta0", rng([ra[t]["etaF (Bayes, feedback) - eta0 (Bayes, no feedback)"][0]
                                                 for t in agents if "etaF (Bayes, feedback) - eta0 (Bayes, no feedback)" in ra[t]]))
        for lab in ("L_U", "L"):
            d = N.get(lab, {})
            for det in ("SelfCheckGPT", "semantic_entropy", "V6", "ctx_lr", "ctx_lr+fb", "V6-prior"):
                put(f"{pre}/{lab}/range/{det}", rng([d[t]["auroc"].get(det) for t in d if t != "pooled"]))
        lcu = N.get("learning_curve", {}).get("L_U", {})
        if lcu:
            for key in next(iter(lcu.values())):
                put(f"{pre}/LC/L_U/mean/{key}", f"{sum(v[key] for v in lcu.values()) / len(lcu):.3f}")
        for tag, d in N.get("channel", {}).items():
            t = TAG.get(tag, tag)
            for k, v in d.items():
                put(f"{pre}/CH/{t}/{k}", f"{v:.2f}" if isinstance(v, float) else v)
    Path(__file__).with_name("numbers.tex").write_text("% generated by make_numbers.py; do not edit\n" + "\n".join(out) + "\n")
    print(f"wrote {len(out)} values")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/pge-paid-wt")
