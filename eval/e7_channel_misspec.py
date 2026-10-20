"""e7b — feedback-channel misspecification (detector-side stress test, $0).

The detector-visible feedback is REPLACED by a signal regenerated from the
reconstructed per-turn truth (analysis only) under an alternative channel.
Eval labels are unchanged (original eval_label). The same channel is used
in train and test (one draw covers every session, then the e6 split).
NO_FEEDBACK_ATTRS stay unobserved in every channel. 3 channel draws per
setting (rng = default_rng([crc32(setting), draw])), each run on split
seeds 42/123/456; AUROC = mean over the 9 (draw, seed) runs; bootstrap
CIs pool all 9 test sets.

V3 / V1 still assume the symmetric eps-flip channel (their eps is chosen
on train labels from the grid 0.05..0.45); nothing is re-tuned per channel.

CAVEAT: the LLM agent's in-context feedback during generation was the
ORIGINAL eps=0.25 channel; only the detector's view is changed here.

Usage: python3 -m eval.e7_channel_misspec [70B] [8B]
Writes results/e7_channel_misspec/summary.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from eval.e7_common import (
    build_dag, channel_rng, channel_stats, diff_cis, fmt_ci, load_corpus, masked,
    mean_auroc, merge_runs, regenerate_channel, run_protocol, with_signals,
)
from pge.baselines.confidence import HighConfContradicted, NegativeSignalFraction
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.posterior import IndependentPosterior
from pge.baselines.selfcheckgpt import SelfCheckGPT

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7_channel_misspec"
N_DRAWS = 3
DETS = ("V3", "V1", "highconf_contradicted", "neg_signal_fraction", "dag_logreg", "selfcheckgpt")
BASELINES = ("highconf_contradicted", "neg_signal_fraction", "dag_logreg", "selfcheckgpt")

SETTINGS = [
    ("recorded (original channel)", None, None),
    ("sym eps=0.10", "sym", 0.10),
    ("sym eps=0.25 (redraw)", "sym", 0.25),
    ("sym eps=0.40", "sym", 0.40),
    ("asym P(-|w)=0.9 P(-|r)=0.35", "asym", (0.9, 0.35)),
    ("asym P(-|w)=0.6 P(-|r)=0.10", "asym", (0.6, 0.10)),
    ("attr eps~U(0.1,0.4)", "attr", (0.1, 0.4)),
    ("recorded + drop 0.3", "drop", 0.3),
    ("recorded + drop 0.5", "drop", 0.5),
    ("lazy user P(-|w)=0.6, no +1", "lazy", 0.6),
]


def factories():
    return [
        ("V3", lambda: IndependentPosterior(by_value=True)),
        ("V1", IndependentPosterior),
        ("highconf_contradicted", HighConfContradicted),
        ("neg_signal_fraction", NegativeSignalFraction),
        ("dag_logreg", DagFeatureClassifier),
        ("selfcheckgpt", SelfCheckGPT),  # channel-independent reference (resamples only)
    ]


def run_corpus(tag, lines):
    trajs, truth = load_corpus(tag)
    labels = np.array([t.eval_label for t in trajs])
    lines += [f"## healthcare {tag} (n={len(trajs)}, hall rate {labels.mean():.3f})", "",
              "| channel | V3 | V1 | HCC | NSF | dag_logreg | SelfCheck | best baseline | V3 - best (item CI) | V3 - best (task CI) | V3 - V1 (item CI) | V3 learned eps (mean) | empirical channel (draw 0) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for name, kind, param in SETTINGS:
        t0 = time.time()
        runs, stats, infos = [], None, []
        for draw in range(1 if kind is None else N_DRAWS):
            if kind is None:
                det_trajs = [masked(t) for t in trajs]
            else:
                sig, info = regenerate_channel(kind, param, trajs, truth, channel_rng(f"{tag}|{name}", draw))
                det_trajs = [with_signals(t, s) for t, s in zip(trajs, sig)]
                infos.append(info)
            if stats is None:
                stats = channel_stats(det_trajs, truth)
            runs.append(run_protocol(det_trajs, labels, factories(), dags=[build_dag(t) for t in det_trajs]))
        R = merge_runs(runs)
        best = max(BASELINES, key=lambda b: mean_auroc(R, b))
        item, clus = diff_cis(R, "V3", best)
        v1, _ = diff_cis(R, "V3", "V1", cluster=False)
        cells = " | ".join(f"{mean_auroc(R, d):.3f}" for d in DETS)
        lines.append(f"| {name.replace('|', chr(92) + '|')} | {cells} | {best} | {fmt_ci(item)} | {fmt_ci(clus)} | {fmt_ci(v1)} | "
                     f"{np.mean(R['eps']['V3']):.3f} | {stats.replace('|', chr(92) + '|')} |")
        detail.append(f"  {name}: V3 per (draw,seed) " + " ".join(f"{x:.3f}" for x in R["auroc"]["V3"])
                      + f" | {best} " + " ".join(f"{x:.3f}" for x in R["auroc"][best])
                      + f" | V3 eps {R['eps']['V3']}")
        if infos and "eps_attr" in infos[0]:
            for d, info in enumerate(infos):
                detail.append(f"    draw {d} eps_attr: " + ", ".join(f"{a}={e:.2f}" for a, e in info["eps_attr"].items()))
        print(f"[{tag}] {name}: V3 {mean_auroc(R, 'V3'):.3f} best {best} {mean_auroc(R, best):.3f} "
              f"{fmt_ci(item)} ({time.time() - t0:.1f}s)", flush=True)
    lines += ["", "Per-run AUROC (draw-major, seeds 42/123/456 within draw):"] + detail + [""]


def main():
    tags = [a for a in sys.argv[1:] if a in ("70B", "8B")] or ["70B", "8B"]
    t0 = time.time()
    lines = [
        "# e7b feedback-channel misspecification (detector-side stress test, offline, $0)",
        "",
        "Detector-visible feedback REPLACED by signals regenerated from reconstructed per-turn truth (analysis",
        "only); eval labels unchanged; same channel in train and test; 3 channel draws x seeds 42/123/456.",
        "CAVEAT: the agent's in-context feedback during generation was the original eps=0.25 channel, so the agent's",
        "behaviour is NOT conditioned on these channels. This tests only the detector's sensitivity to the channel.",
        "'recorded + drop p': the recorded channel with each observed signal set to 0 w.p. p (no truth used).",
        "'lazy user': -1 w.p. 0.6 if wrong, else unobserved; +1 never emitted (correct commitments never confirmed).",
        "V3/V1 assume a symmetric flip channel and pick eps on train labels from 0.05..0.45; not re-tuned per channel.",
        "SelfCheckGPT reads only resamples, so it is channel-independent (identical in every row); included as the",
        "reference that matters on 8B. best baseline = max mean AUROC among HCC/NSF/dag_logreg/SelfCheckGPT, chosen",
        "post hoc (favours the baselines). CIs: paired bootstrap (2000),",
        "all 9 (draw, seed) test sets pooled; item = rows resampled; task = unique task_ids resampled. * = excludes 0.",
        "",
    ]
    for tag in tags:
        run_corpus(tag, lines)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    name = "summary.txt" if len(tags) == 2 else f"summary_{tags[0]}.txt"
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
