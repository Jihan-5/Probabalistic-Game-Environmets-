"""e10_paper_figures — paper figures regenerated from committed result files.

fig_dissociation.pdf  AUROC by corpus for the sampling detector
                      (SelfCheckGPT), the best non-sampling baseline, and
                      the PGE posterior (results/e6_posterior/*.txt)
fig_clarification.pdf hallucination rate and residual confident errors vs
                      questions per session (results/e8_clarification/*.txt)

Usage: python3 -m eval.e10_paper_figures [--out figures/paper]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Validated categorical slots (light mode); aqua < 3:1 contrast, so every
# series is also direct-labeled and marker-coded.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

E6 = {
    "Health\n70B": "healthcare_70B.txt", "Health\n8B": "healthcare_8B.txt",
    "SQL\n70B": "dbbench_70B_pooled.txt", "SQL\n8B": "dbbench_8B.txt",
}
NON_SAMPLING = ("highconf_contradicted", "neg_signal_fraction", "dag_logreg", "gcn")


def mean_auroc(path: Path) -> dict[str, float]:
    out, on = {}, False
    for line in path.read_text().splitlines():
        if line.startswith("## mean over seeds"):
            on = True
            continue
        if on and line.startswith("##"):
            break
        m = re.match(r"\s+(\S+)\s+AUROC ([0-9.]+)", line)
        if on and m:
            out[m.group(1)] = float(m.group(2))
    return out


def fig_dissociation(out: Path):
    groups, vals = list(E6), []
    for g in groups:
        a = mean_auroc(REPO / "results" / "e6_posterior" / E6[g])
        pge = a["pge_posterior_value"] if g.startswith("Health") else a["pge_evidence_nb"]
        vals.append((a["selfcheckgpt"], max(a[k] for k in NON_SAMPLING if k in a), pge))
    series = [("SelfCheckGPT (sampling)", BLUE, "o"),
              ("Best non-sampling baseline", ORANGE, "s"),
              ("PGE posterior (ours)", AQUA, "D")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    w = 0.26
    for j, (name, col, _) in enumerate(series):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(groups))]
        ys = [v[j] for v in vals]
        ax.bar(xs, [y - 0.4 for y in ys], bottom=0.4, width=w, color=col,
               edgecolor="white", linewidth=1.0, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.008, f"{y:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(len(groups) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0.4, 0.84)
    ax.set_ylabel("Session AUROC (mean of 3 splits)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_dissociation.pdf")
    fig.savefig(out / "fig_dissociation.png")
    plt.close(fig)


def read_e8(path: Path) -> dict[str, tuple[float, float, float]]:
    rows = {}
    for line in path.read_text().splitlines():
        m = re.match(r"(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.]", line)
        if m:
            rows[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
    return rows


def fig_clarification(out: Path):
    corpora = [("Healthcare 70B", "70B_42.txt"), ("Healthcare 8B", "8B_42.txt")]
    pols = [("random", "Random", MUTED, "x", "-"),
            ("lowconf", "Lowest confidence", ORANGE, "s", "-"),
            ("posterior", "PGE posterior", BLUE, "o", "-"),
            ("evpi_global", "PGE EVPI (global budget)", AQUA, "D", "-")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), dpi=300, sharex=True)
    for c, (title, fname) in enumerate(corpora):
        R = read_e8(REPO / "results" / "e8_clarification" / fname)
        for r, (metric, idx) in enumerate((("Hallucinating\nsessions", 1), ("Residual confident\nerrors per session", 2))):
            ax = axes[r, c]
            for key, label, col, mk, ls in pols:
                pts = [(0.0, R["none"][idx])]
                pts += sorted((v[0], v[idx]) for k, v in R.items() if k.startswith(key + "@"))
                xs, ys = zip(*pts)
                ax.plot(xs, ys, ls, color=col, marker=mk, ms=3.5, lw=1.4, label=label)
            dq = R["detailed_questions"]
            ax.plot([dq[0]], [dq[idx]], marker="*", ms=9, color=INK, ls="none", label="Detailed Questions")
            ax.annotate("Detailed\nQuestions", (dq[0], dq[idx]), xytext=(6, 4), textcoords="offset points",
                        fontsize=6, color=INK)
            ax.set_xlim(-0.1, 5.2)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(metric, fontsize=7)
            if r == 1:
                ax.set_xlabel("Questions asked per session")
    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_clarification.pdf")
    fig.savefig(out / "fig_clarification.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures" / "paper"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dissociation(out)
    fig_clarification(out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
