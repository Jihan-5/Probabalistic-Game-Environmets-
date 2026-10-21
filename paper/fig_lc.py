"""Figure: test AUROC on L_U vs labelled training sessions, mean over the
confirmatory healthcare agents (thin lines: agents). Same training subsets for
every detector (e11 Amendment 1 scheme). Sources: e16 (V7, intake-LR+F,
ctx-LR+F at 25-200), e11 (V6 curve; V6, ctx-LR+F at all), e12 (intake-LR+F at
all), e16 full-split AUROC (V7 at all). Writes figures/fig_lc.pdf/.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/pge-paid-wt")
R = REPO / "results"
C = json.loads((R / "e11_confirmatory" / "numbers.json").read_text())["learning_curve"]["L_U"]
V = json.loads((R / "e16_v7_confirm" / "numbers.json").read_text())["healthcare"]
X = json.loads((R / "e12_exploratory" / "numbers.json").read_text())
agents = [a for a in V.get("budget", {}) if a in C]
SIZES = [25, 50, 100, 200, "all"]
XS = [25, 50, 100, 200, 420]


def curve(det, a):
    out = []
    for n in SIZES:
        if det == "V6":
            out.append(C[a][f"V6@{n}"])
        elif n == "all":
            out.append({"V7": V["L_U"][a]["auroc"]["V7"], "ctx_lr+fb": C[a]["ctx_lr+fb@all"],
                        "intake_lr+fb": X["learning_curve"][a]["intake_lr+fb@all"]}[det])
        else:
            out.append(V["budget"][a][f"{det}@{n}"])
    return out


INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
SERIES = [("V7", "PGE posterior, intake-aware (V7)", "#2a78d6", "D", "-", 2.4),
          ("intake_lr+fb", "intake-LR+F", "#1baf7a", "^", "--", 1.6),
          ("ctx_lr+fb", "ctx-LR+F", "#eb6834", "s", "-", 1.6),
          ("V6", "PGE posterior (V6)", "#6e7781", "o", "-", 1.6)]
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
for det, label, col, mk, ls, lw in SERIES:
    per = np.array([curve(det, a) for a in agents])
    for row in per:
        ax.plot(XS, row, color=col, lw=0.6, alpha=0.22, zorder=1)
    ax.plot(XS, per.mean(axis=0), color=col, lw=lw, ls=ls, zorder=3, marker=mk, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.8, label=label)
ax.set_xscale("log")
ax.set_xticks(XS)
ax.set_xticklabels(["25", "50", "100", "200", "all (420)"], fontsize=6.5, color=MUTED)
ax.tick_params(axis="y", labelsize=6.5, colors=MUTED)
ax.set_xlabel("Labelled training sessions", fontsize=7, color=INK)
ax.set_ylabel(r"AUROC on $L_U$", fontsize=7, color=INK)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.minorticks_off()
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, labelcolor=INK,
          handlelength=2.8, columnspacing=1.2)
out = Path(__file__).parent / "figures" / "fig_lc"
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", bbox_inches="tight")
print("wrote", out, "agents:", agents)
