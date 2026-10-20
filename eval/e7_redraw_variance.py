"""e7b supplement — how unusual is the RECORDED feedback draw? ($0)

The recorded healthcare channel is ONE draw of the eps=0.25 flip channel.
Here the same channel is redrawn 20 times from reconstructed truth (analysis
only; rng = default_rng([crc32('<tag>|redraw25'), draw])), labels unchanged,
e6 protocol (seeds 42/123/456) per draw. Reports where the recorded channel's
AUROCs and the V3 - best-baseline margin fall in the redraw distribution.

Difference recorded vs redraw has two sources: (i) sampling variance of the
channel noise, (ii) the agent saw the recorded feedback in context, so recorded
signals are coupled to its later commitments; redrawn signals are not. A
coupling diagnostic (next-turn confidence/correctness given the current
turn's truth and recorded signal) is printed per corpus.

Usage: python3 -m eval.e7_redraw_variance
Writes results/e7_channel_misspec/redraw_variance.txt
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eval.e7_common import (
    build_dag, channel_rng, load_corpus, masked, mean_auroc, regenerate_channel,
    run_protocol, with_signals,
)
from pge.baselines.confidence import HighConfContradicted
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.posterior import IndependentPosterior, turn_signal
from pge.baselines.selfcheckgpt import SelfCheckGPT

REPO = Path(__file__).resolve().parent.parent
N_DRAWS = 20
DETS = ("V3", "highconf_contradicted", "dag_logreg", "selfcheckgpt")


def factories():
    return [("V3", lambda: IndependentPosterior(by_value=True)), ("highconf_contradicted", HighConfContradicted),
            ("dag_logreg", DagFeatureClassifier), ("selfcheckgpt", SelfCheckGPT)]


def summarize(name, rec, draws):
    d = np.array(draws)
    pct = 100.0 * np.mean(d <= rec)
    return (f"  {name:22s} recorded {rec:+.3f} | redraws mean {d.mean():+.3f} sd {d.std(ddof=1):.3f} "
            f"min {d.min():+.3f} max {d.max():+.3f} | recorded at percentile {pct:.0f}")


def coupling_lines(tag, trajs, truth):
    """ANALYSIS ONLY: does the agent react to the recorded feedback it saw in context?
    Next-turn outcomes conditioned on (current-turn truth, recorded signal)."""
    out = [f"  agent/feedback coupling on the RECORDED channel ({tag}); next turn t+1 given (w_t, s_t):"]
    for wi in (0, 1):
        for s in (-1, 1):
            nxt = [(w[i + 1], t.turns[i + 1].probability) for t, w in zip(trajs, truth)
                   for i in range(len(t.turns) - 1)
                   if w[i] == wi and turn_signal(t.turns[i]) == s]
            a = np.array(nxt, dtype=float)
            out.append(f"    w_t={wi} s_t={s:+d} n={len(a):4d}  P(w_t+1=1)={a[:, 0].mean():.3f}  "
                       f"P(p_t+1>=0.6)={(a[:, 1] >= 0.6).mean():.3f}  mean p_t+1={a[:, 1].mean():.3f}")
    return out


def main():
    t0 = time.time()
    lines = ["# e7b supplement: recorded feedback draw vs 20 redraws of the same eps=0.25 channel ($0)", "",
             "Mean AUROC over split seeds 42/123/456 per draw. 'margin' = V3 - max(HCC, dag_logreg, SelfCheckGPT)",
             "per draw. Recorded-vs-redraw differences mix channel-noise sampling variance with agent/feedback",
             "coupling (the agent saw the recorded feedback in context; redrawn signals are independent of its",
             "later commitments). The coupling diagnostic below shows the coupling is real on 70B.", ""]
    for tag in ("70B", "8B"):
        trajs, truth = load_corpus(tag)
        y = np.array([t.eval_label for t in trajs])
        rec_trajs = [masked(t) for t in trajs]
        R = run_protocol(rec_trajs, y, factories(), dags=[build_dag(t) for t in rec_trajs])
        rec = {d: mean_auroc(R, d) for d in DETS}
        rec["margin"] = rec["V3"] - max(rec[d] for d in DETS[1:])
        draws = {k: [] for k in rec}
        for draw in range(N_DRAWS):
            sig, _ = regenerate_channel("sym", 0.25, trajs, truth, channel_rng(f"{tag}|redraw25", draw))
            D = [with_signals(t, s) for t, s in zip(trajs, sig)]
            Rd = run_protocol(D, y, factories(), dags=[build_dag(t) for t in D])
            m = {d: mean_auroc(Rd, d) for d in DETS}
            m["margin"] = m["V3"] - max(m[d] for d in DETS[1:])
            for k in m:
                draws[k].append(m[k])
        lines.append(f"## healthcare {tag}")
        for k in ("V3", "highconf_contradicted", "dag_logreg", "margin"):
            lines.append(summarize(k, rec[k], draws[k]))
        lines.append(f"  V3 > best baseline on {sum(x > 0 for x in draws['margin'])}/{N_DRAWS} redraws")
        lines += coupling_lines(tag, trajs, truth)
        lines.append("")
        print(f"[{tag}] done ({time.time() - t0:.0f}s)", flush=True)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    out = REPO / "results" / "e7_channel_misspec" / "redraw_variance.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
