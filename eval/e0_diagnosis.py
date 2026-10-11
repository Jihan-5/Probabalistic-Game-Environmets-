"""e0_diagnosis — zero-cost forensic re-baseline of the pre-rebuild pipeline.

Establishes, from saved results and pure simulation (no LLM calls), the two
failure modes that invalidated the original headline numbers:

  (D1) Structural blindness: with `block_method="position"` and fixed-length
       sequential chains, every trajectory has identical adjacency and block
       assignments, so the graphon log-likelihood varies ONLY through the
       |score| weights — it is a monotone proxy for mean confidence
       magnitude, carrying no structural signal. Demonstrated by direct
       construction + rank correlation ~= 1.

  (D2) Circular evaluation: the high-AUROC detectors in the saved runs
       (neg_cum_evidence ~0.99+, neg_score_fraction ~0.98+) consume behavior
       signals that are truth-matches against the same ground truth that
       defines the eval label. The graphon detector itself sits at or below
       chance in every saved real run.

Usage:  python3 -m eval.e0_diagnosis
Output: results/e0_diagnosis/summary.txt (+ stdout), for RESULTS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_DIR = RESULTS / "e0_diagnosis"

SAVED_RUNS = {
    "dbbench 70B seed42": "main_70B_42.npz",
    "healthcare 70B seed42": "hc_70B_42.npz",
    "healthcare 70B seed123": "hc_70B_123.npz",
    "healthcare 70B seed456": "hc_70B_456.npz",
    "healthcare 70B seed42 score-gated": "hc_70B_42_sgated.npz",
}


def _chain_dag(T: int, scores: np.ndarray) -> ChainOfThoughtDAG:
    """Rebuild a DAG exactly as the old pipeline did: fixed T, edges (i-1, i)."""
    G = ChainOfThoughtDAG()
    for t in range(T):
        G.add_node(
            GuessNode(
                attribute=f"attr_{t}",
                value="v",
                probability=(abs(float(scores[t])) + 1.0) / 2.0,
                score=float(scores[t]),
                turn=t,
            )
        )
        if t > 0:
            G.add_edge(t - 1, t)
    return G


def d1_structural_blindness(T: int = 6, n_traj: int = 500, B: int = 6) -> dict:
    rng = np.random.default_rng(0)
    dags = [_chain_dag(T, rng.uniform(-1, 1, T)) for _ in range(n_traj)]

    est = GraphonEstimator(B=B, block_method="position").fit(dags[: n_traj // 2])

    # (a) identical adjacency + blocks across all trajectories
    A0 = dags[0].adjacency()
    blocks0 = est.assign_blocks(dags[0])
    identical = all(
        np.array_equal(G.adjacency(), A0)
        and np.array_equal(est.assign_blocks(G), blocks0)
        for G in dags
    )

    # (b) graphon LL is a monotone function of mean |score|
    lls = np.array([est.log_likelihood(G) for G in dags])
    mean_abs = np.array([np.abs(G.scores()).mean() for G in dags])
    rho = spearmanr(lls, mean_abs).statistic

    # (c) sign-blindness: flipping every score's sign leaves the LL unchanged
    flipped = [_chain_dag(T, -G.scores()) for G in dags[:50]]
    max_delta = max(
        abs(est.log_likelihood(g) - est.log_likelihood(f))
        for g, f in zip(dags[:50], flipped)
    )

    return {
        "identical_structure": identical,
        "spearman_ll_vs_mean_abs_score": float(rho),
        "max_ll_delta_under_sign_flip": float(max_delta),
    }


def d2_saved_run_table() -> list[str]:
    rows = []
    header = (
        f"{'run':38s} {'graphon_ll':>10s} {'CI':>16s} "
        f"{'neg_cum_ev':>10s} {'neg_frac':>9s} {'n_test':>6s}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for name, fname in SAVED_RUNS.items():
        path = RESULTS / fname
        if not path.exists():
            rows.append(f"{name:38s} (missing: {fname})")
            continue
        d = np.load(path, allow_pickle=True)
        # Two archive layouts exist: multi-detector (auroc_graphon_ll, ...)
        # and single-detector (auroc_point for the graphon LL).
        if "auroc_graphon_ll" in d:
            auroc = float(d["auroc_graphon_ll"])
            lo, hi = float(d["auroc_lo_graphon_ll"]), float(d["auroc_hi_graphon_ll"])
        else:
            auroc = float(d["auroc_point"])
            lo, hi = float(d["auroc_lo"]), float(d["auroc_hi"])
        nce = (
            f"{float(d['auroc_neg_cum_evidence']):.3f}"
            if "auroc_neg_cum_evidence" in d
            else "—"
        )
        nfr = (
            f"{float(d['auroc_neg_score_fraction']):.3f}"
            if "auroc_neg_score_fraction" in d
            else "—"
        )
        n_test = int(d["n_test"]) if "n_test" in d else int(d["test_labels"].shape[0])
        rows.append(
            f"{name:38s} {auroc:10.3f} [{lo:.3f}, {hi:.3f}] "
            f"{nce:>10s} {nfr:>9s} {n_test:6d}"
        )
    return rows


def main() -> None:
    lines = ["# e0_diagnosis — honest re-baseline of the pre-rebuild pipeline", ""]

    d1 = d1_structural_blindness()
    lines += [
        "## D1: structural blindness (position blocks + sequential chains)",
        f"All {500} simulated trajectories share identical adjacency and block "
        f"assignments: {d1['identical_structure']}",
        "Spearman rank correlation between graphon log-likelihood and mean "
        f"|score|: {d1['spearman_ll_vs_mean_abs_score']:.4f} (1.0 = LL is a "
        "pure confidence-magnitude proxy)",
        "Max LL change when flipping the sign of every behavior score: "
        f"{d1['max_ll_delta_under_sign_flip']:.2e} (0 = the detector cannot "
        "distinguish confident-correct from confident-hallucinated)",
        "",
    ]

    lines += ["## D2: saved real-run AUROCs (results/*.npz)", ""]
    lines += d2_saved_run_table()
    lines += [
        "",
        "Reading: the graphon detector is at/below chance on every real run; "
        "the ~0.99 AUROCs come only from neg_cum_evidence / neg_score_fraction, "
        "whose input signal is a truth-match against the same ground truth "
        "that defines the eval label (circular).",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
