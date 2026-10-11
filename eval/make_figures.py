"""make_figures — regenerate paper figures from results/<exp>/ npz caches.

Reads the seed<k>.npz files written by eval.score_trajectories plus the
trajectory JSONL caches (for DAG sizes and the worked example). Headless:
matplotlib Agg backend, PNG at 150 dpi, tight bbox, pure matplotlib.
Every figure is guarded — missing inputs print a skip message, never crash.

Figures:
  figures/<exp>_detector_auroc.png       mean AUROC bars + CI, chance at 0.5
  figures/<exp>_roc_curves.png           ROC curves, top-5 detectors, seed 1
  figures/<exp>_dag_size_stratified.png  top-3 detectors by DAG-size tercile
  figures/worked_example.png             one real hallucinating DAG, annotated

Usage: python3 -m eval.make_figures [--exp e2_dbbench_main] [--paths ...]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from eval.bootstrap_ci import auroc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
DEFAULT_EXP = "e2_dbbench_main"
DEFAULT_PATHS = ["data/trajectories/dbbench_v2/*.jsonl"]
WORKED_EXAMPLE_CACHE = REPO / "data/trajectories/dbbench_v2/70B_42.jsonl"

DPI = 150
# Okabe-Ito subset, fixed order, CVD-validated (validate_palette.js: PASS).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
GRID_KW = dict(color="0.85", linewidth=0.6, zorder=0)

SIGNAL_COLORS = {1: "#009E73", -1: "#D55E00", 0: "#999999"}  # +, -, none


# ------------------------------------------------------------------ loading


def load_seed_npzs(exp: str) -> list[dict]:
    """All seed npz files for an experiment, sorted by seed."""
    def _seed_key(p: Path):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else 0

    out = []
    for f in sorted((REPO / "results" / exp).glob("seed*.npz"), key=_seed_key):
        with np.load(f, allow_pickle=False) as z:
            out.append({k: z[k] for k in z.files})
    return out


def detector_names(npzs: list[dict]) -> list[str]:
    return sorted({k[len("score_"):] for z in npzs for k in z if k.startswith("score_")})


def mean_auroc_ci(npzs: list[dict], names: list[str]) -> dict[str, tuple]:
    """name -> (mean point, mean lower err, mean upper err) over seeds."""
    out = {}
    for name in names:
        pts, los, his = [], [], []
        for z in npzs:
            key = f"metric_{name}"
            if key in z:
                pt, lo, hi, _ = z[key]
                pts.append(pt), los.append(pt - lo), his.append(hi - pt)
        if pts:
            out[name] = (
                float(np.nanmean(pts)),
                float(np.nanmean(los)),
                float(np.nanmean(his)),
            )
    return out


def _save(fig, path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------- fig 1: AUROC bars


def fig_detector_auroc(exp: str, npzs: list[dict]) -> None:
    names = detector_names(npzs)
    stats = mean_auroc_ci(npzs, names)
    if not stats:
        print(f"[skip] {exp}_detector_auroc: no metric_* keys in npz files")
        return
    ranked = sorted(stats.items(), key=lambda kv: kv[1][0])  # ascending: best on top
    labels = [k for k, _ in ranked]
    pts = np.array([v[0] for _, v in ranked])
    err = np.array([[v[1] for _, v in ranked], [v[2] for _, v in ranked]])

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(labels) + 1.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, pts, height=0.62, color=PALETTE[0], zorder=3)
    for b in bars:  # thin rounded-feel: keep flat but subtle edge
        b.set_edgecolor("white")
        b.set_linewidth(0.5)
    ax.errorbar(
        pts, y, xerr=err, fmt="none", ecolor="0.25", elinewidth=1.1,
        capsize=2.5, zorder=4,
    )
    ax.axvline(0.5, color="0.4", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        0.5, 1.005, "chance", color="0.4", fontsize=8, ha="center",
        va="bottom", transform=ax.get_xaxis_transform(),
    )
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("AUROC (mean over seeds, 95% bootstrap CI)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"{exp}: detector AUROC")
    ax.xaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, FIG_DIR / f"{exp}_detector_auroc.png")


# --------------------------------------------------------- fig 2: ROC curves


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(float)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - y) / n_neg])
    return fpr, tpr


def fig_roc_curves(exp: str, npzs: list[dict], top_k: int = 5) -> None:
    if not npzs:
        print(f"[skip] {exp}_roc_curves: no npz files")
        return
    z = npzs[0]  # first seed
    labels = z.get("labels")
    if labels is None:
        print(f"[skip] {exp}_roc_curves: npz missing labels")
        return
    stats = mean_auroc_ci(npzs, detector_names(npzs))
    top = [n for n, _ in sorted(stats.items(), key=lambda kv: -kv[1][0])][:top_k]
    top = [n for n in top if f"score_{n}" in z]
    if not top:
        print(f"[skip] {exp}_roc_curves: no scores for top detectors in first seed")
        return

    linestyles = ["-", "--", "-.", ":", (0, (5, 1))]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.9, zorder=1)
    for i, name in enumerate(top):
        rc = _roc_curve(z[f"score_{name}"], labels)
        if rc is None:
            continue
        fpr, tpr = rc
        a = auroc(z[f"score_{name}"], labels)
        ax.plot(
            fpr, tpr, color=PALETTE[i % len(PALETTE)],
            linestyle=linestyles[i % len(linestyles)], linewidth=1.8,
            label=f"{name} ({a:.2f})", zorder=3,
        )
    seed = int(z["seed"]) if "seed" in z else 0
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{exp}: ROC, top {len(top)} detectors (seed {seed})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, FIG_DIR / f"{exp}_roc_curves.png")


# ------------------------------------------- fig 3: DAG-size stratified AUROC


def _dag_sizes_for(z: dict, paths: list[str]) -> np.ndarray | None:
    """Per-test-row DAG sizes: prefer the npz's own column, else rebuild a
    task_id -> size map from the trajectory caches."""
    if "dag_sizes" in z:
        return np.asarray(z["dag_sizes"], dtype=int)
    if "task_ids" not in z:
        return None
    try:
        from eval.score_trajectories import build_dag, load_trajectories
    except ImportError:
        return None
    trajs = load_trajectories(paths)
    if not trajs:
        return None
    size_by_task: dict[str, int] = {}
    for t in trajs:
        size_by_task.setdefault(t.task_id, len(build_dag(t)))
    return np.array([size_by_task.get(str(tid), -1) for tid in z["task_ids"]])


def fig_dag_size_stratified(exp: str, npzs: list[dict], paths: list[str]) -> None:
    if not npzs:
        print(f"[skip] {exp}_dag_size_stratified: no npz files")
        return
    stats = mean_auroc_ci(npzs, detector_names(npzs))
    top3 = [n for n, _ in sorted(stats.items(), key=lambda kv: -kv[1][0])][:3]
    if not top3:
        print(f"[skip] {exp}_dag_size_stratified: no detectors")
        return

    sizes_all = np.concatenate(
        [s for s in (_dag_sizes_for(z, paths) for z in npzs) if s is not None]
        or [np.array([])]
    )
    sizes_all = sizes_all[sizes_all >= 0]
    if sizes_all.size == 0:
        print(f"[skip] {exp}_dag_size_stratified: could not recover DAG sizes")
        return
    q1, q2 = np.quantile(sizes_all, [1 / 3, 2 / 3])
    if len(np.unique(sizes_all)) < 2 or q1 == q2 == sizes_all.max():
        print(
            f"[skip] {exp}_dag_size_stratified: DAG sizes are (near-)constant "
            f"(unique sizes: {sorted(np.unique(sizes_all).tolist())})"
        )
        return

    def tercile(s):
        return np.where(s <= q1, 0, np.where(s <= q2, 1, 2))

    terc_labels = [
        f"small (n≤{q1:.0f})",
        f"medium ({q1:.0f}<n≤{q2:.0f})",
        f"large (n>{q2:.0f})",
    ]

    # Pool AUROC per (detector, tercile) over seeds.
    vals = {name: [[] for _ in range(3)] for name in top3}
    for z in npzs:
        sizes = _dag_sizes_for(z, paths)
        if sizes is None or "labels" not in z:
            continue
        terc = tercile(sizes)
        for name in top3:
            key = f"score_{name}"
            if key not in z:
                continue
            for t in range(3):
                m = (terc == t) & (sizes >= 0)
                y = z["labels"][m]
                if m.sum() >= 4 and 0 < y.sum() < len(y):
                    vals[name][t].append(auroc(z[key][m], y))

    width = 0.8 / len(top3)
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, name in enumerate(top3):
        means = [np.nanmean(v) if v else np.nan for v in vals[name]]
        ax.bar(
            x + (i - (len(top3) - 1) / 2) * width, means, width * 0.92,
            color=PALETTE[i], label=name, zorder=3, edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks(x, terc_labels, fontsize=9)
    ax.set_ylabel("AUROC (mean over seeds)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{exp}: top-3 detectors by DAG-size tercile")
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, FIG_DIR / f"{exp}_dag_size_stratified.png")


# ------------------------------------------------- fig 4: worked example DAG


def _pick_worked_example(cache: Path):
    """First eval_label==1 trajectory with >=1 negative-aggregate-signal turn."""
    from pge.trajectory import load_jsonl

    for traj in load_jsonl(cache):
        if traj.eval_label != 1:
            continue
        if any(sum(t.signals.values()) < 0 for t in traj.turns):
            return traj
    return None


def fig_worked_example(cache: Path = WORKED_EXAMPLE_CACHE) -> None:
    if not cache.exists():
        print(f"[skip] worked_example: cache {cache} missing")
        return
    traj = _pick_worked_example(cache)
    if traj is None:
        print(f"[skip] worked_example: no hallucinating trajectory with a "
              f"negative signal in {cache}")
        return
    try:
        from eval.score_trajectories import build_dag
        dag = build_dag(traj)
    except Exception as e:
        print(f"[skip] worked_example: DAG rebuild failed ({e})")
        return

    n = len(traj.turns)
    xs = np.arange(n, dtype=float)
    ys = np.zeros(n)

    fig, ax = plt.subplots(figsize=(2.6 * n, 3.4))

    # Directed edges: straight for adjacent turns, arcs ABOVE the nodes for
    # skips (labels live below; negative rad bends the arc upward here).
    for p, c in dag.edges:
        rad = 0.0 if c - p == 1 else -(0.22 + 0.10 * (c - p))
        arrow = FancyArrowPatch(
            (xs[p], ys[p]), (xs[c], ys[c]),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14,
            color="0.45", linewidth=1.2, zorder=2,
            shrinkA=26, shrinkB=26,
        )
        ax.add_patch(arrow)

    # Nodes colored by aggregate signal.
    for i, turn in enumerate(traj.turns):
        s = int(np.sign(sum(turn.signals.values())))
        ax.scatter(
            xs[i], ys[i], s=2200, color=SIGNAL_COLORS[s],
            edgecolor="white", linewidth=2.0, zorder=3,
        )
        val = turn.guess_value
        if len(val) > 16:
            val = val[:14] + "…"
        ax.annotate(
            f"{turn.attribute}\n{val} p={turn.probability:.2f}",
            (xs[i], ys[i]), xytext=(0, -46), textcoords="offset points",
            ha="center", va="top", fontsize=8.5, zorder=4,
        )

    # Legend: signal semantics (color + explicit text, never color alone).
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SIGNAL_COLORS[k], markeredgecolor="white",
                   label=lbl)
        for k, lbl in ((1, "signal +1 (supported)"),
                       (-1, "signal −1 (contradicted)"),
                       (0, "signal 0 (unobserved)"))
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    ax.set_title(
        f"Worked example: {traj.task_id} ({traj.domain}, eval_label=1)",
        fontsize=10,
    )
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-1.2, 1.6)
    ax.axis("off")
    _save(fig, FIG_DIR / "worked_example.png")


# -------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default=DEFAULT_EXP)
    ap.add_argument("--paths", nargs="+", default=DEFAULT_PATHS)
    args = ap.parse_args(argv)

    npzs = load_seed_npzs(args.exp)
    if not npzs:
        print(f"[skip] no npz files under results/{args.exp}/ "
              "(run eval.score_trajectories first)")
    else:
        fig_detector_auroc(args.exp, npzs)
        fig_roc_curves(args.exp, npzs)
        fig_dag_size_stratified(args.exp, npzs, args.paths)
    fig_worked_example()


if __name__ == "__main__":
    main()
