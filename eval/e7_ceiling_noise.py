"""e7c — oracle ceiling vs feedback noise, and how much of it V3 recovers ($0).

For eps in {0.00, 0.05, ..., 0.45} the detector-visible feedback is
REPLACED by a symmetric flip channel regenerated from reconstructed per-turn
truth (analysis only); labels are the original eval_label; 3 channel draws
per eps (rng = default_rng([crc32(setting), draw])).

  oracle   e6_ceiling's ceiling: exact label-event posterior with priors
           estimated IN-SAMPLE from true per-turn errors by
           (attribute, value, bucket) and the TRUE eps. Optimistic. Reported
           on the full corpus (as e6) and on the 3 test splits (same
           in-sample priors) for comparability with the learned detectors.
  V3/V1, HighConfContradicted, dag_logreg: learned, e6 protocol (fit on
           train, score test, seeds 42/123/456) on the regenerated channel.
           V3's eps grid is fixed at 0.05..0.45, so at eps=0 it cannot pick
           the true value (fixed a priori, not changed here).

Sanity check: the oracle on the RECORDED channel must reproduce
results/e6_posterior/ceiling.txt (0.824 70B, 0.817 8B).

Usage: python3 -m eval.e7_ceiling_noise [70B] [8B]
Writes results/e7_ceiling_noise/{summary.txt, table_<tag>.csv, ceiling_vs_noise.png}
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.e7_common import (
    SEEDS, auroc, build_dag, channel_rng, diff_cis, fmt_ci, load_corpus, masked, mean_auroc,
    merge_runs, regenerate_channel, run_protocol, with_signals,
)
from eval.score_trajectories import split_task_ids
from pge.baselines.confidence import HighConfContradicted
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.posterior import (
    HIGH, IndependentPosterior, conf_bucket, emission, label_threshold,
    poisson_binomial_tail, turn_signal,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "e7_ceiling_noise"
EPS_LEVELS = tuple(np.round(np.arange(0.0, 0.451, 0.05), 2))
N_DRAWS = 3
DETS = ("V3", "V1", "highconf_contradicted", "dag_logreg", "selfcheckgpt")
SHORT = {"highconf_contradicted": "HCC", "dag_logreg": "dag", "selfcheckgpt": "SelfCheck"}


def factories():
    return [
        ("V3", lambda: IndependentPosterior(by_value=True)),
        ("V1", IndependentPosterior),
        ("highconf_contradicted", HighConfContradicted),
        ("dag_logreg", DagFeatureClassifier),
        ("selfcheckgpt", SelfCheckGPT),  # channel-independent reference
    ]


def oracle_priors(trajs, truth):
    """ANALYSIS ONLY: in-sample error rate by (attribute, value, bucket)."""
    num, den = defaultdict(float), defaultdict(float)
    for t, w in zip(trajs, truth):
        for tt, wi in zip(t.turns, w):
            k = (tt.attribute, tt.guess_value, conf_bucket(tt.probability))
            num[k] += wi
            den[k] += 1
    return {k: float(np.clip(num[k] / den[k], 0.01, 0.99)) for k in den}


def oracle_scores(det_trajs, pi, eps):
    out = []
    for t in det_trajs:
        q = []
        for tt in t.turns:
            if tt.probability < HIGH:
                continue
            p0, s = pi[(tt.attribute, tt.guess_value, conf_bucket(tt.probability))], turn_signal(tt)
            l1, l0 = emission(s, 1, eps), emission(s, 0, eps)
            q.append(p0 * l1 / (p0 * l1 + (1 - p0) * l0))
        out.append(poisson_binomial_tail(np.array(q), label_threshold(len(q))) if q else 0.0)
    return np.array(out)


def test_split_auroc(scores, labels, ids):
    v = []
    for seed in SEEDS:
        _, te_ids = split_task_ids(ids, 0.7, seed)
        te = np.array([i for i, t in enumerate(ids) if t in te_ids])
        v.append(auroc(scores[te], labels[te]))
    return float(np.mean(v))


def run_corpus(tag, lines):
    trajs, truth = load_corpus(tag)
    labels = np.array([t.eval_label for t in trajs])
    ids = [t.task_id for t in trajs]
    pi = oracle_priors(trajs, truth)
    rec = oracle_scores([masked(t) for t in trajs], pi, 0.25)
    lines += [f"## healthcare {tag} (n={len(trajs)}, hall rate {labels.mean():.3f})",
              f"sanity: oracle on RECORDED channel (eps=0.25) full-corpus AUROC {auroc(rec, labels):.3f} "
              f"(e6_ceiling: 0.824 70B / 0.817 8B); test-split mean {test_split_auroc(rec, labels, ids):.3f}", "",
              "| eps | oracle (full, in-sample) | oracle (test splits) | V3 | V1 | HCC | dag_logreg | V3 gap to oracle(test) | SelfCheck | V3 - best(HCC,dag,SelfCheck) item CI | task CI | V3 learned eps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["eps,oracle_full,oracle_test,V3,V1,highconf_contradicted,dag_logreg,selfcheckgpt,V3_eps_mean"]
    rows = []
    for eps in EPS_LEVELS:
        t0 = time.time()
        of, ot, A, E, runs = [], [], defaultdict(list), [], []
        for draw in range(N_DRAWS):
            sig, _ = regenerate_channel("sym", float(eps), trajs, truth, channel_rng(f"{tag}|ceil|{eps:.2f}", draw))
            det_trajs = [with_signals(t, s) for t, s in zip(trajs, sig)]
            o = oracle_scores(det_trajs, pi, float(eps))
            of.append(auroc(o, labels))
            ot.append(test_split_auroc(o, labels, ids))
            R = run_protocol(det_trajs, labels, factories(), dags=[build_dag(t) for t in det_trajs])
            runs.append(R)
            for d in DETS:
                A[d].append(mean_auroc(R, d))
            E.extend(R["eps"]["V3"])
        m = {d: float(np.mean(A[d])) for d in DETS}
        bname = max(("highconf_contradicted", "dag_logreg", "selfcheckgpt"), key=lambda d: m[d])
        item, clus = diff_cis(merge_runs(runs), "V3", bname)
        rows.append((float(eps), np.mean(of), np.mean(ot), m, np.mean(E)))
        lines.append(f"| {eps:.2f} | {np.mean(of):.3f} | {np.mean(ot):.3f} | {m['V3']:.3f} | {m['V1']:.3f} | "
                     f"{m['highconf_contradicted']:.3f} | {m['dag_logreg']:.3f} | {np.mean(ot) - m['V3']:+.3f} | "
                     f"{m['selfcheckgpt']:.3f} | {fmt_ci(item)} ({SHORT[bname]}) | {fmt_ci(clus)} | {np.mean(E):.3f} |")
        csv.append(f"{eps:.2f},{np.mean(of):.4f},{np.mean(ot):.4f},{m['V3']:.4f},{m['V1']:.4f},"
                   f"{m['highconf_contradicted']:.4f},{m['dag_logreg']:.4f},{m['selfcheckgpt']:.4f},{np.mean(E):.4f}")
        print(f"[{tag}] eps={eps:.2f} oracle {np.mean(of):.3f}/{np.mean(ot):.3f} V3 {m['V3']:.3f} "
              f"HCC {m['highconf_contradicted']:.3f} dag {m['dag_logreg']:.3f} ({time.time() - t0:.1f}s)", flush=True)
    lines.append("")
    (OUT / f"table_{tag}.csv").write_text("\n".join(csv) + "\n")
    return rows


def plot(all_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional
        return f"(no PNG: matplotlib unavailable: {e})"
    series = [("oracle (test splits)", lambda r: r[2], "#1f1f1f", "--"),
              ("V3 (learned)", lambda r: r[3]["V3"], "#2a6fdb", "-"),
              ("V1", lambda r: r[3]["V1"], "#7fa7e8", "-"),
              ("HighConfContradicted", lambda r: r[3]["highconf_contradicted"], "#d9822b", "-"),
              ("dag_logreg", lambda r: r[3]["dag_logreg"], "#8c8c8c", "-"),
              ("SelfCheckGPT (channel-free)", lambda r: r[3]["selfcheckgpt"], "#5aa469", ":")]
    fig, axes = plt.subplots(1, len(all_rows), figsize=(5.2 * len(all_rows), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (tag, rows) in zip(axes, all_rows.items()):
        x = [r[0] for r in rows]
        for label, f, c, ls in series:
            ax.plot(x, [f(r) for r in rows], ls, color=c, marker="o", ms=3, lw=1.6, label=label)
        ax.axvline(0.25, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_title(f"healthcare {tag}", fontsize=10)
        ax.set_xlabel("symmetric flip rate eps (regenerated channel)")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC (mean of 3 draws x 3 split seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = OUT / "ceiling_vs_noise.png"
    fig.savefig(path, dpi=160)
    return f"PNG: {path.relative_to(REPO)}"


def main():
    tags = [a for a in sys.argv[1:] if a in ("70B", "8B")] or ["70B", "8B"]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# e7c oracle ceiling vs symmetric feedback noise (offline, $0)",
        "",
        "Feedback REPLACED by a symmetric flip channel regenerated from reconstructed truth (analysis only),",
        "3 draws per eps; labels = original eval_label. Oracle = in-sample (attribute, value, bucket) priors from",
        "truth + true eps (optimistic ceiling, as e6_ceiling). Learned detectors: e6 protocol on the regenerated",
        "channel. All AUROCs are means over 3 draws (x 3 split seeds for learned detectors). SelfCheckGPT reads",
        "only resamples (channel-independent reference). 'best' = max mean AUROC of HCC/dag_logreg/SelfCheckGPT",
        "per eps (post hoc). CIs: paired bootstrap 2000 over the 9 pooled test sets; item / task-cluster. * excl. 0.",
        "CAVEAT: the agent's in-context feedback was the original eps=0.25 channel (detector-side test only).",
        "",
    ]
    all_rows = {tag: run_corpus(tag, lines) for tag in tags}
    lines.append(plot(all_rows))
    lines.append(f"runtime {time.time() - t0:.0f}s")
    name = "summary.txt" if len(tags) == 2 else f"summary_{tags[0]}.txt"
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
