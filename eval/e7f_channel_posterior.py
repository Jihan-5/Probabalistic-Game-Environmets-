"""e7f — general-channel posterior (V6) under every e7b channel ($0).

Motivation (logged in RESULTS.md): under the e7b "lazy user" channel
(contradicts wrong commitments w.p. 0.6, never confirms correct ones) the
symmetric-channel posterior V3 collapses (70B 0.425, 8B 0.576) while
dag_logreg holds (0.829 / 0.797). V6 (pge.baselines.posterior.ChannelPosterior)
replaces the symmetric flip channel with free contradiction rates
(r0, r1) chosen on train labels and confirmation rates (g0, g1) fit by
least squares. Same channels, draws, seeds and CIs as e7b; V6 is fixed
before seeing these results on any channel other than the recorded one.

Usage: python3 -m eval.e7f_channel_posterior [70B] [8B]
Writes results/e7f_channel_posterior/summary[_<tag>].txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from eval.e7_channel_misspec import N_DRAWS, SETTINGS
from eval.e7_common import (
    build_dag, channel_rng, diff_cis, fmt_ci, load_corpus, masked, mean_auroc,
    merge_runs, regenerate_channel, run_protocol, with_signals,
)
from pge.baselines.confidence import HighConfContradicted, NegativeSignalFraction
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.posterior import ChannelPosterior, IndependentPosterior

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7f_channel_posterior"
DETS = ("V6", "V3", "highconf_contradicted", "neg_signal_fraction", "dag_logreg")
BASELINES = ("highconf_contradicted", "neg_signal_fraction", "dag_logreg")


class _V6(ChannelPosterior):
    """Records the learned channel for the report."""

    channels: list = []

    def fit(self, *a, **k):
        super().fit(*a, **k)
        _V6.channels.append((float(self.r[0]), float(self.r[1]), self.g[0], self.g[1]))
        return self


def factories():
    return [
        ("V6", _V6),
        ("V3", lambda: IndependentPosterior(by_value=True)),
        ("highconf_contradicted", HighConfContradicted),
        ("neg_signal_fraction", NegativeSignalFraction),
        ("dag_logreg", DagFeatureClassifier),
    ]


def run_corpus(tag, lines):
    trajs, truth = load_corpus(tag)
    labels = np.array([t.eval_label for t in trajs])
    lines += [f"## healthcare {tag} (n={len(trajs)}, hall rate {labels.mean():.3f})", "",
              "| channel | V6 | V3 | HCC | NSF | dag_logreg | best baseline | V6 - best (item CI) | V6 - best (task CI) | V6 - V3 (item CI) | learned (r0, r1, g0, g1), mean |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, kind, param in SETTINGS:
        t0 = time.time()
        _V6.channels = []
        runs = []
        for draw in range(1 if kind is None else N_DRAWS):
            if kind is None:
                det = [masked(t) for t in trajs]
            else:
                sig, _ = regenerate_channel(kind, param, trajs, truth, channel_rng(f"{tag}|{name}", draw))
                det = [with_signals(t, s) for t, s in zip(trajs, sig)]
            runs.append(run_protocol(det, labels, factories(), dags=[build_dag(t) for t in det]))
        R = merge_runs(runs)
        best = max(BASELINES, key=lambda b: mean_auroc(R, b))
        item, clus = diff_cis(R, "V6", best)
        v3, _ = diff_cis(R, "V6", "V3", cluster=False)
        ch = np.mean(np.array(_V6.channels), axis=0)
        cells = " | ".join(f"{mean_auroc(R, d):.3f}" for d in DETS)
        lines.append(f"| {name} | {cells} | {best} | {fmt_ci(item)} | {fmt_ci(clus)} | {fmt_ci(v3)} | "
                     f"({ch[0]:.2f}, {ch[1]:.2f}, {ch[2]:.2f}, {ch[3]:.2f}) |")
        print(f"[{tag}] {name}: V6 {mean_auroc(R, 'V6'):.3f} V3 {mean_auroc(R, 'V3'):.3f} "
              f"best {best} {mean_auroc(R, best):.3f} ({time.time() - t0:.0f}s)", flush=True)
    lines.append("")


def main():
    tags = [a for a in sys.argv[1:] if a in ("70B", "8B")] or ["70B", "8B"]
    t0 = time.time()
    lines = ["# e7f general-channel posterior V6 under the e7b channels (offline, $0)", "",
             "Same protocol, channels, draws and CIs as e7b. V6: free contradiction rates (r0, r1) chosen on",
             "train labels (grid r0 0..0.45, r1 0.30..0.95, r1 > r0 + 0.1), confirmation rates by least squares.",
             "CAVEAT (as e7b): the agent's in-context feedback was the original channel; detector-side test only.", ""]
    for tag in tags:
        run_corpus(tag, lines)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    name = "summary.txt" if len(tags) == 2 else f"summary_{tags[0]}.txt"
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
