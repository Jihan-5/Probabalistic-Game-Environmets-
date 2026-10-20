"""e7_common — shared harness for the e7 offline stress tests ($0, no API calls).

Protocol is e6_posterior's, unchanged: hash split by task_id
(split_task_ids(ids, 0.7, seed), seeds 42/123/456), fit on train, score the
test split once, AUROC + AUPRC, paired bootstrap of AUROC differences with
the per-seed test sets pooled (item-level, eval.e6_posterior.paired_boot_diff,
n=2000, rng seed 0). A task-cluster bootstrap (resample unique task_ids, all
pooled rows of a task together) is added as a second, more conservative CI.

TRUTH USE (analysis only). Per-turn correctness is reconstructed from the
generator seed stream (eval.e6_ceiling.load_with_truth). It is used ONLY to
  (a) define alternative eval labels (E7a), and
  (b) regenerate an alternative detector-visible feedback channel for the
      detector-side stress tests (E7b/E7c).
Detectors only ever receive cloned trajectories whose eval_label is masked
to 0, plus train-split labels through fit(). They never see per-turn truth.
"""

from __future__ import annotations

import dataclasses
import zlib
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e6_ceiling import load_with_truth
from eval.e6_posterior import paired_boot_diff
from eval.score_trajectories import average_precision, build_dag, split_task_ids
from pge.baselines.posterior import turn_signal
from pge.benchmarks.healthcare_v2 import ATTRIBUTES_V2, NO_FEEDBACK_ATTRS
from pge.trajectory import load_jsonl

REPO = Path(__file__).resolve().parent.parent
SEEDS = (42, 123, 456)
TRAIN_FRAC = 0.7
CORPORA = {"70B": ("70B_42", 600, 42), "8B": ("8B_42", 200, 42)}


# ------------------------------------------------------------------ data


def load_corpus(tag: str):
    """(trajectories, per-turn truth arrays). Truth is ANALYSIS-ONLY."""
    name, n, seed = CORPORA[tag]
    D = load_with_truth(name, n, seed)
    # Reorder to cache-file order so pooled bootstrap resamples match e6 exactly.
    order = [t.task_id for t in load_jsonl(REPO / "data/trajectories/healthcare_v2" / f"{name}.jsonl")]
    by_id = {t.task_id: (t, w) for t, w in D}
    D = [by_id[tid] for tid in order]
    return [t for t, _ in D], [np.array(w, dtype=int) for _, w in D]


def masked(traj):
    """Detector-facing copy: eval_label masked to 0 (labels go via fit only)."""
    return dataclasses.replace(traj, eval_label=0)


def with_signals(traj, signals):
    """Detector-facing copy with a replaced feedback channel (label masked)."""
    turns = [dataclasses.replace(tt, signals={"noisy_feedback": int(s)})
             for tt, s in zip(traj.turns, signals)]
    return dataclasses.replace(traj, turns=turns, eval_label=0)


def label_from_truth(traj, w, high: float, frac: float) -> int:
    """eval_label_v2 rule with (confidence_threshold=high, error_fraction=frac)."""
    hi = [int(wi) for tt, wi in zip(traj.turns, w) if tt.probability >= high]
    if not hi:
        return 0
    return int(sum(hi) / len(hi) >= frac)


# --------------------------------------------------------------- channels


def channel_rng(setting: str, draw: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(setting.encode()), draw])


def regenerate_channel(kind: str, param, trajs, truth, rng):
    """Per-turn {-1,0,+1} signals for every session under an alternative
    channel. NO_FEEDBACK_ATTRS stay unobserved (0) in every channel.

    sym   param=eps         s = (-1 if wrong else +1), flipped w.p. eps
    asym  param=(a, b)      P(-1|wrong)=a, P(-1|right)=b, else +1
    attr  param=(lo, hi)    sym with eps_attr ~ U(lo, hi), fixed per attribute
                            within a draw (redrawn per draw)
    drop  param=p           RECORDED channel, each observed signal -> 0 w.p. p
                            (no truth used)
    lazy  param=c           -1 w.p. c if wrong, else 0; never reacts to a
                            correct commitment (+1 never emitted)
    Returns (signals, info) where info records any per-draw parameters.
    """
    info = {}
    eps_attr = None
    if kind == "attr":
        lo, hi = param
        eps_attr = {a: float(rng.uniform(lo, hi)) for a in sorted(ATTRIBUTES_V2)}
        info["eps_attr"] = eps_attr
    out = []
    for t, w in zip(trajs, truth):
        sig = []
        for tt, wi in zip(t.turns, w):
            if tt.attribute in NO_FEEDBACK_ATTRS:
                sig.append(0)
                continue
            u = rng.random()
            if kind == "sym":
                base = -1 if wi else 1
                sig.append(-base if u < param else base)
            elif kind == "asym":
                a, b = param
                sig.append(-1 if u < (a if wi else b) else 1)
            elif kind == "attr":
                base = -1 if wi else 1
                sig.append(-base if u < eps_attr[tt.attribute] else base)
            elif kind == "drop":
                s = turn_signal(tt)
                sig.append(0 if (s != 0 and u < param) else s)
            elif kind == "lazy":
                sig.append(-1 if (wi and u < param) else 0)
            else:
                raise ValueError(kind)
        out.append(sig)
    return out, info


def channel_stats(trajs, truth) -> str:
    """Empirical P(-1|wrong), P(-1|right), P(+1|...), observed fraction."""
    c = np.zeros((2, 3))  # [wrong][s+1]
    for t, w in zip(trajs, truth):
        for tt, wi in zip(t.turns, w):
            c[int(wi), turn_signal(tt) + 1] += 1
    pw, pr = c[1] / c[1].sum(), c[0] / c[0].sum()
    return (f"P(-1|wrong)={pw[0]:.3f} P(+1|wrong)={pw[2]:.3f} "
            f"P(-1|right)={pr[0]:.3f} P(+1|right)={pr[2]:.3f} "
            f"observed={(c[:, 0].sum() + c[:, 2].sum()) / c.sum():.3f}")


# ---------------------------------------------------------------- harness


def run_protocol(trajs, labels, factories, dags=None, seeds=SEEDS):
    """e6 protocol. factories: [(name, zero-arg constructor)].

    Returns dict with per-seed auroc/auprc lists, pooled test scores per
    detector, pooled y and pooled task ids (seed order), learned eps lists.
    """
    labels = np.asarray(labels, dtype=int)
    if dags is None:
        dags = [build_dag(t) for t in trajs]
    ids = [t.task_id for t in trajs]
    R = {"auroc": {}, "auprc": {}, "scores": {}, "eps": {}, "y": [], "tid": []}
    for seed in seeds:
        tr_ids, te_ids = split_task_ids(ids, TRAIN_FRAC, seed)
        tr = [i for i, t in enumerate(ids) if t in tr_ids]
        te = [i for i, t in enumerate(ids) if t in te_ids]
        y_te = labels[te]
        for name, make in factories:
            det = make()
            det.fit([trajs[i] for i in tr], [dags[i] for i in tr], labels[tr])
            s = np.array([det.score(trajs[i], dags[i]) for i in te], dtype=float)
            R["auroc"].setdefault(name, []).append(auroc(s, y_te))
            R["auprc"].setdefault(name, []).append(average_precision(s, y_te))
            R["scores"].setdefault(name, []).append(s)
            if getattr(det, "eps", None) is not None:
                R["eps"].setdefault(name, []).append(float(det.eps))
        R["y"].append(y_te)
        R["tid"].append(np.array([ids[i] for i in te]))
    return R


def merge_runs(runs):
    """Pool several run_protocol results (e.g. channel draws) into one."""
    M = {"auroc": {}, "auprc": {}, "scores": {}, "eps": {}, "y": [], "tid": []}
    for R in runs:
        for key in ("auroc", "auprc", "scores", "eps"):
            for n, v in R[key].items():
                M[key].setdefault(n, []).extend(v)
        M["y"].extend(R["y"])
        M["tid"].extend(R["tid"])
    return M


def pooled(R, name):
    return np.concatenate(R["scores"][name])


def mean_auroc(R, name) -> float:
    return float(np.nanmean(R["auroc"][name]))


def mean_auprc(R, name) -> float:
    return float(np.nanmean(R["auprc"][name]))


def cluster_boot_diff(sa, sb, y, groups, n=2000, seed=0):
    """Bootstrap CI for AUROC(sa) - AUROC(sb), resampling unique group ids
    (all pooled rows of a sampled group enter together)."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    G = len(members)
    diffs = []
    for _ in range(n):
        b = np.concatenate([members[g] for g in rng.integers(0, G, G)])
        yb = y[b]
        if yb.sum() in (0, len(b)):
            continue
        diffs.append(auroc(sa[b], yb) - auroc(sb[b], yb))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def diff_cis(R, a, b, cluster=True):
    """(item-level CI, task-cluster CI) for AUROC(a) - AUROC(b), pooled."""
    sa, sb = pooled(R, a), pooled(R, b)
    y = np.concatenate(R["y"])
    item = paired_boot_diff(sa, sb, y)
    clus = cluster_boot_diff(sa, sb, y, np.concatenate(R["tid"])) if cluster else None
    return item, clus


def fmt_ci(ci) -> str:
    if ci is None:
        return "n/a"
    m, lo, hi = ci
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


__all__ = [
    "SEEDS", "CORPORA", "load_corpus", "masked", "with_signals", "label_from_truth",
    "channel_rng", "regenerate_channel", "channel_stats", "run_protocol", "merge_runs",
    "pooled", "mean_auroc", "mean_auprc", "cluster_boot_diff", "diff_cis", "fmt_ci",
    "build_dag", "auroc",
]
