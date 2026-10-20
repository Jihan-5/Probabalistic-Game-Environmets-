"""Figure: test AUROC on L_U vs number of labelled training sessions (Amendment 1).

Mean over the confirmatory healthcare agents (each agent: mean over split
seeds and 5 random training subsets per size). One axis, three series in fixed
colours; thin agent-level lines behind each mean. Writes figures/fig_lc.pdf/.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/pge-paid-wt")
N = json.loads((REPO / "results" / "e11_confirmatory" / "numbers.json").read_text())
LC = N["learning_curve"]["L_U"]
SIZES = [25, 50, 100, 200, "all"]
X = [25, 50, 100, 200, 420]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
SERIES = [("V6", "PGE posterior (V6)", "#2a78d6", "D"), ("ctx_lr+fb", "ctx-LR+F", "#eb6834", "s"),
          ("ctx_lr", "ctx-LR (no reactions)", "#6e7781", "o")]

fig, ax = plt.subplots(figsize=(3.4, 2.15), dpi=200)
for det, label, col, mk in SERIES:
    per = np.array([[LC[a][f"{det}@{n}"] for n in SIZES] for a in LC])
    for row in per:
        ax.plot(X, row, color=col, lw=0.6, alpha=0.25, zorder=1)
    m = per.mean(axis=0)
    ax.plot(X, m, color=col, lw=2, zorder=3, marker=mk, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.8, label=label)
ax.set_xscale("log")
ax.set_xticks(X)
ax.set_xticklabels(["25", "50", "100", "200", "all (420)"], fontsize=6.5, color=MUTED)
ax.tick_params(axis="y", labelsize=6.5, colors=MUTED)
ax.set_xlabel("Labelled training sessions", fontsize=7, color=INK)
ax.set_ylabel(r"AUROC on $L_U$", fontsize=7, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.minorticks_off()
ax.legend(fontsize=6, frameon=False, loc="lower right", labelcolor=INK)
fig.tight_layout()
out = Path(__file__).parent / "figures" / "fig_lc"
fig.savefig(str(out) + ".pdf")
fig.savefig(str(out) + ".png")
print("wrote", out, "agents:", list(LC))
