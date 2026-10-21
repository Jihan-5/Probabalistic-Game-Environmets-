"""Write numbers.tex: \\setval{key}{value} for every number the paper cites.

Sources (repo): results/e11_confirmatory/numbers.json (confirmatory, prefix C)
and dev_numbers.json (development, prefix D). A key the paper uses but this
file does not define prints as a red ?? in the PDF, so no number can be typed
by hand.

Usage: python3 make_numbers.py /path/to/repo
"""

import json
import math

import numpy as np
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
    # a nonzero bound that rounds to zero keeps a fourth decimal, so a starred
    # interval never prints as "+.000"
    s = f"{x:+.4f}" if 0 < abs(x) < 0.0005 else f"{x:+.3f}"
    return s.replace("+0.", "+.").replace("-0.", "$-$.")


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
    xf = Path(repo) / "results" / "e12_exploratory" / "numbers.json"
    if xf.exists():
        X = json.loads(xf.read_text())
        for lab in ("L_U", "L"):
            for tag, d in X.get(lab, {}).items():
                if tag == "pooled":
                    for q, c in d.items():
                        put(f"X/{lab}/pooled/{q}", ci(c) if " - " in q else f"{a3(c[0])}\\,[{a3(c[1])}, {a3(c[2])}]")
                    continue
                for det, v in d.items():
                    put(f"X/{lab}/{TAG.get(tag, tag)}/{det}", a3(v))
                    put(f"t:X/{lab}/{TAG.get(tag, tag)}/{det}", a3(v, lead=False))
            vals = [d.get("intake_lr+fb") for t, d in X.get(lab, {}).items() if t != "pooled"]
            if vals:
                put(f"X/{lab}/range/intake_lr+fb", f"{min(vals):.3f}--{max(vals):.3f}")
        xlc = X.get("learning_curve", {})
        cf = Path(repo) / "results" / "e11_confirmatory" / "numbers.json"
        clc = json.loads(cf.read_text()).get("learning_curve", {}).get("L_U", {}) if cf.exists() else {}
        if xlc:
            for key in next(iter(xlc.values())):
                put(f"X/LC/L_U/mean/{key}", f"{sum(v[key] for v in xlc.values()) / len(xlc):.3f}")
            for size in (25, 50, 100, 200, "all"):
                k = [t for t in xlc if t in clc]
                lead = [t for t in k if clc[t][f"V6@{size}"] > xlc[t][f"intake_lr+fb@{size}"]]
                put(f"X/derived/LC/lead_vs_intake/{size}", f"{len(lead)} of {len(k)}")
        pcq = X.get("percommit_q", {})
        for tag, d in pcq.items():
            for k in ("auroc_q", "ece_q", "mean_q", "wrong"):
                put(f"X/PCQ/{TAG.get(tag, tag)}/{k}", a3(d[k]))
        for k in ("auroc_prior", "auroc_reaction", "auroc_disagree", "auroc_conf"):
            v = [d[k] for d in pcq.values() if k in d]
            if v:
                put(f"X/PCQ/range/{k}", f"{min(v):.3f}--{max(v):.3f}")
        for k in ("auroc_q", "ece_q"):
            if pcq:
                put(f"X/PCQ/range/{k}", f"{min(d[k] for d in pcq.values()):.3f}--{max(d[k] for d in pcq.values()):.3f}")
    vf = Path(repo) / "results" / "e16_v7_confirm" / "numbers.json"
    if vf.exists():
        V = json.loads(vf.read_text())
        for dname, short in (("healthcare", "hc"), ("travel", "tr")):
            D = V.get(dname, {})
            for lab in ("L_U", "L"):
                for tag, d in D.get(lab, {}).items():
                    if tag == "pooled":
                        for q, c in d.items():
                            put(f"V/{short}/{lab}/pooled/{q}", ci(c) if " - " in q else f"{a3(c[0])}\\,[{a3(c[1])}, {a3(c[2])}]")
                        continue
                    t = TAG.get(tag, tag)
                    for det, v in d["auroc"].items():
                        put(f"V/{short}/{lab}/{t}/{det}", a3(v))
                        put(f"t:V/{short}/{lab}/{t}/{det}", a3(v, lead=False))
                    for q, c in d["diffs"].items():
                        put(f"V/{short}/{lab}/{t}/{q}", ci(c))
                ags = [t for t in D.get(lab, {}) if t != "pooled"]
                if ags:
                    for q in D[lab][ags[0]]["diffs"]:
                        cis = [D[lab][t]["diffs"][q] for t in ags]
                        put(f"V/{short}/{lab}/npos/{q}", f"{sum(c[1] > 0 for c in cis)} of {len(cis)}")
                        put(f"V/{short}/{lab}/nneg/{q}", f"{sum(c[2] < 0 for c in cis)} of {len(cis)}")
                    vals = [D[lab][t]["auroc"]["V7"] for t in ags]
                    put(f"V/{short}/{lab}/range/V7", f"{min(vals):.3f}--{max(vals):.3f}")
            bud = D.get("budget", {})
            if bud:
                keys = next(iter(bud.values())).keys()
                for k in keys:
                    put(f"V/{short}/budget/mean/{k}", f"{sum(v[k] for v in bud.values()) / len(bud):.3f}")
                for size in (25, 50, 100, 200):
                    for other in ("intake_lr+fb", "ctx_lr+fb"):
                        ks = [t for t, v in bud.items() if f"V7@{size}" in v]
                        put(f"V/{short}/budget/lead_vs_{other}/{size}",
                            f"{sum(bud[t][f'V7@{size}'] > bud[t][f'{other}@{size}'] for t in ks)} of {len(ks)}")
            # share of V6's gap to the oracle V6-session that V7 closes (means over agents)
            cf = Path(repo) / "results" / "e11_confirmatory" / "numbers.json"
            if cf.exists() and short == "hc":
                C = json.loads(cf.read_text())
                ags = [t for t in D.get("L_U", {}) if t != "pooled" and t in C.get("rules_aware", {})]
                if ags:
                    v6 = np.mean([D["L_U"][t]["auroc"]["V6"] for t in ags])
                    v7m = np.mean([D["L_U"][t]["auroc"]["V7"] for t in ags])
                    vs = np.mean([C["rules_aware"][t]["V6-session"] for t in ags])
                    put("V/hc/derived/gap_closed", f"{100 * (v7m - v6) / (vs - v6):.0f}")
                    put("V/hc/derived/mean_V6session", f"{vs:.3f}")
            pc = D.get("percommit_v7", {})
            if pc:
                for k in ("auroc_q", "ece_q"):
                    put(f"V/{short}/PCQ/range/{k}", f"{min(d[k] for d in pc.values()):.3f}--{max(d[k] for d in pc.values()):.3f}")
    pf = Path(repo) / "results" / "e19_percommit" / "numbers.json"
    if pf.exists():
        PC = json.loads(pf.read_text())
        for which, d in PC.items():
            for tag, m in d.items():
                t = TAG.get(tag, tag)
                for k, v in m.items():
                    if k.startswith("ece"):
                        put(f"PC/{which}/{t}/{k}", a3(v[0]))  # point only: bootstrap ECE is biased upward
                    elif " - " in k:
                        put(f"PC/{which}/{t}/{k}", ci(v))
                    else:
                        put(f"PC/{which}/{t}/{k}", f"{a3(v[0])}\\,[{a3(v[1])}, {a3(v[2])}]")
                        put(f"pt:PC/{which}/{t}/{k}", a3(v[0]))
            ags = [t for t in d if t != "pooled"]
            for k in d[ags[0]] if ags else []:
                vals = [d[t][k][0] for t in ags]
                put(f"PC/{which}/range/{k}", f"{min(vals):.3f}--{max(vals):.3f}")
    cf2 = Path(repo) / "results" / "e11_confirmatory" / "numbers.json"
    CLR = json.loads(cf2.read_text()).get("clarification", {}) if cf2.exists() else {}
    if CLR:
        put("CL/n", CLR.get("n_sessions"))
        for c, d in CLR.items():
            if isinstance(d, dict) and "questions" in d:
                for k, v in d.items():
                    put(f"CL/{c}/{k}", f"{v:.2f}" if k == "questions" else a3(v))
        for q, c in CLR.get("diffs", {}).items():
            put(f"CL/d/{q}", ci(c))
    c7 = Path(repo) / "results" / "e21_clarification_v7" / "numbers.json"
    if c7.exists():
        C7 = json.loads(c7.read_text())
        for c, d in C7["conditions"].items():
            for k, v in d.items():
                put(f"CL7/{c}/{k}", f"{v:.2f}" if k == "questions" else a3(v))
        for q, d in C7["comparisons"].items():
            for k, v in d.items():
                put(f"CL7/d/{k}/{q}", ci(v))
    jd = Path(repo) / "results" / "e20_judge" / "numbers.json"
    if jd.exists():
        JD = json.loads(jd.read_text())
        for lab in ("L_U", "L"):
            for tag, d in JD.get(lab, {}).items():
                if tag == "pooled":
                    for q, c in d.items():
                        put(f"JD/{lab}/pooled/{q}", ci(c) if " - " in q else f"{a3(c[0])}\\,[{a3(c[1])}, {a3(c[2])}]")
                    continue
                for det, v in d["auroc"].items():
                    put(f"JD/{lab}/{TAG.get(tag, tag)}/{det}", a3(v))
                    put(f"t:JD/{lab}/{TAG.get(tag, tag)}/{det}", a3(v, lead=False))
                for q, c in d["diffs"].items():
                    put(f"JD/{lab}/{TAG.get(tag, tag)}/{q}", ci(c))
            ags = [t for t in JD.get(lab, {}) if t != "pooled"]
            if ags:
                cis = [JD[lab][t]["diffs"]["V7 - llm_judge"] for t in ags]
                put(f"JD/{lab}/npos/V7 - llm_judge", f"{sum(c[1] > 0 for c in cis)} of {len(cis)}")
        pcj = JD.get("percommit", {})
        if pcj:
            put("JD/PC/range/judge", f"{min(d['judge'] for d in pcj.values()):.3f}--{max(d['judge'] for d in pcj.values()):.3f}")
    jf = Path(repo) / "results" / "e18_joint_bootstrap" / "numbers.json"
    if jf.exists():
        J = json.loads(jf.read_text())
        for lab, d in J.items():
            for q, v in d.items():
                jc = v["joint"]
                put(f"J/{lab}/{q}", ci(jc) if " - " in q else f"{a3(jc[0])}\\,[{a3(jc[1])}, {a3(jc[2])}]")
                if q.startswith("H1: AUROC(SelfCheckGPT)"):
                    jse = d.get("H1: AUROC(semantic_entropy)", {}).get("joint")
                    inb = all(0.40 <= x[1] and x[2] <= 0.60 for x in (jc, jse) if x)
                    put(f"J/{lab}/H1band", "both still inside the band" if inb else
                        "at least one end now outside the band, so the registered criterion holds under the registered bootstrap but not under this stricter one")
                hp = v.get("p_holm")
                if hp is not None and hp == hp:
                    put(f"J/{lab}/holm/{q}", "$<$.001" if hp < 0.001 else f"{hp:.3f}".replace("0.", ".", 1))
    sf = Path(repo) / "results" / "e17_strong_baselines" / "numbers.json"
    if sf.exists():
        S7 = json.loads(sf.read_text())
        for q, c in S7.get("pooled", {}).items():
            put(f"S/pooled/{q}", ci(c))
        ags = [t for t in S7 if t not in ("pooled", "budget")]
        for t in ags:
            for det, v in S7[t]["auroc"].items():
                put(f"S/{TAG.get(t, t)}/{det}", a3(v))
                put(f"t:S/{TAG.get(t, t)}/{det}", a3(v, lead=False))
            for q, c in S7[t]["diffs"].items():
                put(f"S/{TAG.get(t, t)}/{q}", ci(c))
        if ags:
            for q in S7[ags[0]]["diffs"]:
                cis = [S7[t]["diffs"][q] for t in ags]
                put(f"S/npos/{q}", f"{sum(c[1] > 0 for c in cis)} of {len(cis)}")
                put(f"S/nneg/{q}", f"{sum(c[2] < 0 for c in cis)} of {len(cis)}")
            for det in S7[ags[0]]["auroc"]:
                put(f"S/mean/{det}", f"{np.mean([S7[t]['auroc'][det] for t in ags]):.3f}")
        pv = [S7.get("pooled", {}).get(f"V7 - {d}", [None])[0] for d in ("intake_lr+fb", "intake_lr+fb (cv)", "intake_gbm+fb")]
        pv = [v for v in pv if v is not None]
        if pv:
            put("S/range/vs_raw", f"{min(pv):.2f}--{max(pv):.2f}")
        bud = S7.get("budget", {})
        vf2 = Path(repo) / "results" / "e16_v7_confirm" / "numbers.json"
        vb = json.loads(vf2.read_text())["healthcare"].get("budget", {}) if vf2.exists() else {}
        if bud:
            for size in (25, 50, 100, 200):
                k = f"dep_lr+fb (cv)@{size}"
                vals = [d[k] for d in bud.values() if k in d]
                if vals:
                    put(f"S/budget/mean/{k}", f"{np.mean(vals):.3f}")
                ks = [t for t in bud if t in vb and k in bud[t]]
                put(f"S/budget/v7_lead/{size}", f"{sum(vb[t][f'V7@{size}'] > bud[t][k] for t in ks)} of {len(ks)}")
    ef5 = Path(repo) / "results" / "e15_regimes_confirm_draws5" / "numbers.json"
    ef = ef5 if ef5.exists() else Path(repo) / "results" / "e15_regimes_confirm" / "numbers.json"
    if ef.exists():
        E = json.loads(ef.read_text())
        regimes = next(iter(E.values())).keys()
        for rg in regimes:
            first = next(iter(E.values()))[rg]
            dets = (first[0] if isinstance(first, list) else first).keys()
            for det in dets:
                # mean over agents of the per-agent mean over draws; sd over draws of the agent mean
                per = []
                for t in E:
                    if rg not in E[t]:
                        continue
                    rows = E[t][rg] if isinstance(E[t][rg], list) else [E[t][rg]]
                    per.append([r[det] for r in rows])
                k = min(len(x) for x in per)
                agent_mean_by_draw = np.mean([x[:k] for x in per], axis=0)
                put(f"E/mean/{rg}/{det}", f"{np.mean(agent_mean_by_draw):.3f}")
                if k > 1:
                    put(f"E/sd/{rg}/{det}", f"{np.std(agent_mean_by_draw):.3f}")
            put("E/draws", str(k))
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
        if pc:
            put(f"{pre}/PC/maxdiff_e", f"{max(abs(d['e1'] - d['e0']) for d in pc.values()):.2f}")
        ags_s = [t for t in N.get("L_U", {}) if t != "pooled"]
        sv = [N["L_U"][t]["auroc"].get(d) for t in ags_s for d in ("gcn", "dag_logreg", "best graphon")]
        sv = [v for v in sv if v is not None]
        if sv:
            put(f"{pre}/L_U/range/structure", f"{min(sv):.2f}--{max(sv):.2f}")
        ra, lu = N.get("rules_aware", {}), N.get("L_U", {})
        agents = [t for t in ra if t != "pooled" and t in lu]
        put(f"{pre}/derived/V6session_gap/range",
            rng([ra[t].get("V6-session", 0) - lu[t]["auroc"].get("V6", 0) for t in agents]))
        for k in ("eta0 (Bayes, no feedback)", "etaF (Bayes, feedback)"):
            put(f"{pre}/RA/range/{k}", rng([ra[t].get(k) for t in agents]))
        put(f"{pre}/RA/range/etaF - eta0", rng([ra[t]["etaF (Bayes, feedback) - eta0 (Bayes, no feedback)"][0]
                                                 for t in agents if "etaF (Bayes, feedback) - eta0 (Bayes, no feedback)" in ra[t]]))
        for q in ("LR-session+fb - LR-session", "etaF (Bayes, feedback) - etaX (Bayes, shown reactions)"):
            v = [ra[t][q][0] for t in agents if q in ra[t]]
            if v:
                put(f"{pre}/RA/range/{q}", f"{sd(min(v))} to {sd(max(v))}")
        for lab in ("L_U", "L"):
            d = N.get(lab, {})
            for det in ("SelfCheckGPT", "semantic_entropy", "V6", "ctx_lr", "ctx_lr+fb", "V6-prior"):
                put(f"{pre}/{lab}/range/{det}", rng([d[t]["auroc"].get(det) for t in d if t != "pooled"]))
        # counts over agents, so "every agent" is never typed by hand
        WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        NAME = {"70B": "Llama-70B", "8B": "Llama-8B", "qwen72b": "Qwen-72B", "gpt-4o-mini": "GPT-4o-mini",
                "mistral-small-24b": "Mistral-24B", "gpt-4o-mini (travel)": "GPT-4o-mini", "8B (travel)": "Llama-8B"}
        for lab in ("L_U", "L"):
            ags = [t for t in N.get(lab, {}) if t != "pooled"]
            if not ags:
                continue
            put(f"{pre}/derived/{lab}/nagents", WORD.get(len(ags), len(ags)))
            for det in ("SelfCheckGPT", "semantic_entropy"):
                v = {t: N[lab][t]["auroc"].get(det) for t in ags if N[lab][t]["auroc"].get(det) is not None}
                if v:
                    put(f"{pre}/derived/{lab}/argmax/{det}", NAME.get(max(v, key=v.get), max(v, key=v.get)))
            for q in N[lab][ags[0]].get("diffs", {}):
                cis = {t: N[lab][t]["diffs"][q] for t in ags if q in N[lab][t].get("diffs", {})}
                pos = [t for t, c in cis.items() if c[1] > 0]
                neg = [t for t, c in cis.items() if c[2] < 0]
                ns = [t for t in cis if t not in pos and t not in neg]
                put(f"{pre}/derived/{lab}/npos/{q}", f"{len(pos)} of {len(cis)}")
                put(f"{pre}/derived/{lab}/nneg/{q}", f"{len(neg)} of {len(cis)}")
                put(f"{pre}/derived/{lab}/ns_agents/{q}", ", ".join(NAME.get(t, t) for t in ns) or "none")
        for lab, dd in N.get("learning_curve", {}).items():
            for size in (25, 50, 100, 200, "all"):
                k = [t for t, d in dd.items() if f"V6@{size}" in d]
                lead = [t for t in k if dd[t][f"V6@{size}"] > dd[t][f"ctx_lr+fb@{size}"]]
                put(f"{pre}/derived/LC/{lab}/lead/{size}", f"{len(lead)} of {len(k)}")
        lcu = N.get("learning_curve", {}).get("L_U", {})
        if lcu:
            for key in next(iter(lcu.values())):
                put(f"{pre}/LC/L_U/mean/{key}", f"{sum(v[key] for v in lcu.values()) / len(lcu):.3f}")
        for tag, d in N.get("channel", {}).items():
            t = TAG.get(tag, tag)
            for k, v in d.items():
                put(f"{pre}/CH/{t}/{k}", f"{v:.2f}" if isinstance(v, float) else v)
    # point-estimate twins ("pt:" + key) for the abstract and the contributions
    import re as _re
    pts = []
    for line in out:
        m = _re.match(r"\\setval\{([^}]*)\}\{(.*)\}$", line)
        if m and "\\,[" in m.group(2) and not m.group(1).startswith(("t:", "pt:")):
            pts.append(f"\\setval{{pt:{m.group(1)}}}{{{m.group(2).split(chr(92) + ',[')[0]}}}")
            star = "$^{*}$" if m.group(2).endswith("$^{*}$") else ""
            pts.append(f"\\setval{{s:{m.group(1)}}}{{{m.group(2).split(chr(92) + ',[')[0]}{star}}}")
    out += pts
    Path(__file__).with_name("numbers.tex").write_text("% generated by make_numbers.py; do not edit\n" + "\n".join(out) + "\n")
    print(f"wrote {len(out)} values")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/pge-paid-wt")
