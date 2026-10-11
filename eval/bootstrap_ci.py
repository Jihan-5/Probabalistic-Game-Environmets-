"""AUROC + bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, mannwhitneyu


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC where higher `scores` -> more likely positive class (label=1).

    Returns nan if either class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    U = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) AUROC and percentile CI.

    Stratified bootstrap: resample positives and negatives separately so each
    bootstrap sample has the same class proportions as the original.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        point = auroc(scores, labels)
        return point, float("nan"), float("nan")

    point = auroc(scores, labels)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        boot[i] = auroc(scores[idx], labels[idx])
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, lo, hi


def wilcoxon_one_sided(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """One-sided Mann-Whitney U test: H1 = scores_pos > scores_neg."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    res = mannwhitneyu(scores_pos, scores_neg, alternative="greater")
    return float(res.pvalue)
