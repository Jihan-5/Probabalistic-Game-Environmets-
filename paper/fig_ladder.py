"""Figure: where the detection signal is, per agent (AUROC on L_U).

Dot plot, one row per agent, on a single AUROC axis. Series (fixed order and
colours, never cycled): SelfCheckGPT (model only), ctx-LR+F and V6 (practical
detectors on PGE observables), eta0 and etaF (exact Bayes references; oracle
information). Reads results/e11_confirmatory/numbers.json (or dev_numbers.json
with --dev). Writes figures/fig_ladder.pdf/.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path("/home/claude/pge-paid-wt")
DEV = "--dev" in sys.argv
N = json.loads((REPO / "results" / "e11_confirmatory" / ("dev_numbers.json" if DEV else "numbers.json")).read_text())
AGENTS = ([("70B (dev)", "Llama-70B (dev)"), ("8B (dev)", "Llama-8B (dev)")] if DEV else
          [("70B", "Llama-70B"), ("8B", "Llama-8B"), ("qwen72b", "Qwen-72B"), ("gpt-4o-mini", "GPT-4o-mini"),
           ("mistral-small-24b", "Mistral-24B")])
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
SERIES = [  # (label, getter, colour, marker, filled)
    ("SelfCheckGPT", lambda t: N["L_U"][t]["auroc"].get("SelfCheckGPT"), "#6e7781", "o", True),
    ("ctx-LR+F", lambda t: N["L_U"][t]["auroc"].get("ctx_lr+fb"), "#eb6834", "s", True),
    ("PGE posterior (V6)", lambda t: N["L_U"][t]["auroc"].get("V6"), "#2a78d6", "D", True),
    ("V6-session (oracle event)", lambda t: N["rules_aware"][t].get("V6-session"), "#2a78d6", "D", False),
    (r"Population risk ($\bar\eta$)", lambda t: N["rules_aware"][t].get("eta0 (Bayes, no feedback)"), INK, "o", False),
    (r"Bayes, with reactions ($\eta_F$)", lambda t: N["rules_aware"][t].get("etaF (Bayes, feedback)"), INK, "o", True),
]

agents = [(k, lab) for k, lab in AGENTS if k in N.get("L_U", {}) and k in N.get("rules_aware", {})]
fig, ax = plt.subplots(figsize=(3.4, 0.42 * len(agents) + 1.0), dpi=200)
for i, (key, lab) in enumerate(agents):
    y = len(agents) - 1 - i
    vals = [g(key) for _, g, *_ in SERIES]
    xs = [v for v in vals if v is not None]
    ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    for (name, _, col, mk, filled), v in zip(SERIES, vals):
        if v is None:
            continue
        ax.scatter([v], [y], s=34, marker=mk, zorder=3, facecolor=col if filled else "white",
                   edgecolor="white" if filled else col, linewidth=0.9 if filled else 1.3)
ax.set_yticks(range(len(agents)))
ax.set_yticklabels([lab for _, lab in reversed(agents)], fontsize=7, color=INK)
ax.set_xlim(0.4, 1.0)
ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
ax.text(0.505, len(agents) - 0.45, "chance", fontsize=6, color=MUTED, va="center")
ax.set_xlabel(r"Session AUROC on the user-specific label $L_U$", fontsize=7, color=INK)
ax.tick_params(axis="x", labelsize=6.5, colors=MUTED)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
ax.set_ylim(-0.6, len(agents) - 0.3)
handles = [plt.Line2D([], [], ls="", marker=mk, markersize=5, markerfacecolor=col if filled else "white",
                      markeredgecolor="white" if filled else col, markeredgewidth=0.9 if filled else 1.3, label=name)
           for name, _, col, mk, filled in SERIES]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.6, frameon=False, bbox_to_anchor=(0.55, 1.0),
           handletextpad=0.3, columnspacing=0.9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (0.42 * len(agents) + 1.0)))
out = Path(__file__).parent / "figures" / ("fig_ladder_dev" if DEV else "fig_ladder")
fig.savefig(str(out) + ".pdf")
fig.savefig(str(out) + ".png")
print("wrote", out)
