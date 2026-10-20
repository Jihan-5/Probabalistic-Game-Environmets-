"""Audit the empirical feedback channel of an LLM-patient corpus ($0, offline).

ANALYSIS ONLY. Reconstructs per-turn noise-free truth from the generator
seed stream (sample_session_v2(default_rng(seed), i), task_id hcv2_<i>) and
reports P(signal | agent wrong / right) overall, per attribute, per persona
and per confidence band, plus extractor/agent parse failures and the symmetric
flip rate the channel corresponds to. Truth never feeds a detector.

Usage: python3 -m eval.paid.audit_llmpatient <jsonl> [--seed 42] [--out results/...txt]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pge.benchmarks.healthcare_llmpatient import SIGNAL
from pge.benchmarks.healthcare_v2 import NO_FEEDBACK_ATTRS, sample_session_v2
from pge.trajectory import load_jsonl


def with_truth(trajs, seed: int):
    n = 1 + max(int(re.search(r"(\d+)$", t.task_id).group(1)) for t in trajs)
    rng = np.random.default_rng(seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(n)}
    rows = []
    for t in trajs:
        s = sessions[t.task_id]
        assert [tt.attribute for tt in t.turns] == s.schedule, f"{t.task_id}: session stream mismatch"
        audit = {a["turn"]: a for a in t.meta.get("turn_audit", [])}
        for tt in t.turns:
            a = audit.get(tt.turn, {})
            rows.append({
                "attr": tt.attribute,
                "wrong": int(tt.guess_value != s.user_state[tt.attribute]),
                "sig": int(tt.signals.get(SIGNAL, 0)),
                "p": tt.probability,
                "persona": t.meta.get("persona", "?"),
                "observed": a.get("extractor_status") not in (None, "no_feedback_attr"),
                "ex_failed": a.get("extractor_status") == "failed",
                "agent_failed": str(a.get("agent_parse_status", "")).endswith("failed"),
                "verdict": bool(a.get("patient_verdict_word")),
                "label": t.eval_label,
            })
    return rows


def channel(rows) -> dict:
    out = {}
    for w in (1, 0):
        r = [x for x in rows if x["wrong"] == w]
        n = len(r)
        out[w] = (n, *(sum(x["sig"] == s for x in r) / n if n else float("nan") for s in (-1, 0, 1)))
    return out


def fmt(name, rows) -> str:
    c = channel(rows)
    n1, m1, z1, p1 = c[1]
    n0, m0, z0, p0 = c[0]
    return (f"  {name:26s} n_wrong={n1:5d} P(-1|w)={m1:.3f} P(0|w)={z1:.3f} P(+1|w)={p1:.3f} | "
            f"n_right={n0:5d} P(-1|r)={m0:.3f} P(0|r)={z0:.3f} P(+1|r)={p0:.3f} | "
            f"P(0)={np.mean([x['sig'] == 0 for x in rows]):.3f}")


def report(path: str, seed: int) -> str:
    trajs = load_jsonl(path)
    rows = with_truth(trajs, seed)
    obs = [x for x in rows if x["observed"]]
    labels = np.array([t.eval_label for t in trajs])
    L = [f"# audit_llmpatient: {path} (n={len(trajs)} sessions, {len(rows)} turns, seed {seed}; "
         f"hall rate {labels.mean():.3f})", "",
         "ANALYSIS ONLY (truth reconstructed from the seed stream).", "",
         f"agent parse failures: {sum(x['agent_failed'] for x in rows)}/{len(rows)}; "
         f"extractor failures (-> signal 0): {sum(x['ex_failed'] for x in obs)}/{len(obs)}; "
         f"patient replies with verdict words: {sum(x['verdict'] for x in obs)}/{len(obs)}", "",
         "## channel on turns with a patient reply (w = agent wrong vs noise-free truth)",
         fmt("ALL observed", obs), fmt("ALL turns (incl. no-reply)", rows)]
    c = channel(obs)
    p_detect, p_false = c[1][1], c[0][1]
    L.append(f"  contradiction lift P(-1|w)/P(-1|r) = {p_detect / max(p_false, 1e-9):.2f}; "
             f"v2 reference (eps=0.25, symmetric): P(-1|w)=0.75, P(-1|r)=0.25, P(0)=0")
    acc = [x for x in obs if x["sig"] != 0]
    if acc:
        flip = np.mean([(x["sig"] < 0) != bool(x["wrong"]) for x in acc])
        L.append(f"  among nonzero signals: disagreement with truth = {flip:.3f} (v2: eps = 0.25)")
    L += ["", "## per attribute (observed turns)"]
    by = defaultdict(list)
    for x in obs:
        by[x["attr"]].append(x)
    for a in sorted(by):
        L.append(fmt(a, by[a]))
    L += ["", "## per persona (observed turns)"]
    byp = defaultdict(list)
    for x in obs:
        byp[x["persona"]].append(x)
    for p in sorted(byp):
        L.append(fmt(p, byp[p]))
    L += ["", "## per confidence band (observed turns)"]
    for lo, hi in ((0, 0.6), (0.6, 0.75), (0.75, 1.01)):
        L.append(fmt(f"p in [{lo}, {hi})", [x for x in obs if lo <= x["p"] < hi]))
    if any(x["attr"] in NO_FEEDBACK_ATTRS and x["observed"] for x in rows):
        L.append("\n(note: corpus was generated with --feedback-on-all)")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = report(args.path, args.seed)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
