"""e8_clarification — posterior-guided clarification vs Detailed Questions ($0).

Closed-loop question: given a budget of user questions, which commitments
should the system ask the user to confirm? Offline simulation on cached
healthcare-v2 trajectories:

  * Asking about attribute a reveals its true value (the user knows it), so
    that commitment becomes correct.
  * NO DILUTION: the outcome label is eval_label_v2's rule computed over the
    session's ORIGINAL confident commitments (p >= 0.6) only. A confirmed
    commitment keeps its original confidence, so a question can only help
    by fixing a wrong confident commitment, never by adding a correct item
    to the denominator. (An earlier draft set p := 1 on confirmation; that
    let questions about irrelevant commitments flip labels by dilution, so
    it was discarded.) Also reported: residual confident errors per session.
  * No cascade: a corrected parent does not re-derive its children. This is
    applied identically to every policy, so the comparison is fair and the
    measured benefit is a lower bound.

Per-turn truth is reconstructed from the generator seed stream (analysis and
outcome simulation only). Policies never see truth or labels; they see only
detector-visible fields and a V3 posterior fit on the train split.

Policies (test split, seeds 42/123/456):
  none                 ask nothing
  detailed_questions   ask every underdetermined attribute in the session
                       (upfront Detailed-Questions baseline; burden varies)
  detailed_all         ask every attribute (max burden)
  random@b             b random commitments per session
  lowconf@b            the b lowest-confidence commitments
  posterior@b          the b commitments with highest posterior error q_t
  evpi@b               greedy: the commitment whose confirmation most
                       reduces P(label event), b times
  evpi_global@B        one test-set-wide budget, spent greedily on the
                       largest expected reductions across ALL sessions
                       (so detection decides WHICH sessions get questions)
  oracle@b             the b wrong high-confidence commitments (upper bound)

Usage: python3 -m eval.e8_clarification [--corpus 70B_42|8B_42]
Writes results/e8_clarification/<corpus>.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.e6_ceiling import load_with_truth
from eval.score_trajectories import split_task_ids
from pge.baselines.posterior import (
    IndependentPosterior,
    emission,
    label_threshold,
    poisson_binomial_tail,
    turn_signal,
)
from pge.benchmarks.healthcare_v2 import UNDERDETERMINED

REPO = Path(__file__).resolve().parent.parent
SEEDS = (42, 123, 456)
BUDGETS = (1, 2, 3, 4, 5)
HIGH, FRAC = 0.6, 0.5
N_CORPUS = {"70B_42": 600, "8B_42": 200}


# ------------------------------------------------------------ simulation


def residual(traj, wrong, asked) -> tuple[int, int]:
    """(label, residual confident errors) after confirming `asked`; the
    confident set is fixed to the original p >= HIGH commitments."""
    n_high = n_wrong = 0
    for i, turn in enumerate(traj.turns):
        if turn.probability >= HIGH:
            n_high += 1
            n_wrong += wrong[i] and i not in asked
    if n_high == 0:
        return 0, 0
    return int(n_wrong / n_high >= FRAC), n_wrong


def confident(traj) -> list[int]:
    return [j for j, t in enumerate(traj.turns) if t.probability >= HIGH]


# ------------------------------------------------------------ posterior view


def turn_posteriors(det: IndependentPosterior, traj) -> np.ndarray:
    """Posterior error q_t for EVERY turn (detector-visible inputs only)."""
    q = []
    for turn in traj.turns:
        pi = det._prior(turn)
        s = turn_signal(turn)
        l1, l0 = emission(s, 1, det.eps), emission(s, 0, det.eps)
        q.append(pi * l1 / (pi * l1 + (1 - pi) * l0))
    return np.array(q)


def event_prob(traj, q, asked) -> float:
    """P(label event | evidence) after confirming `asked` (q_t -> 0 on the
    original confident set)."""
    qs = []
    for i, turn in enumerate(traj.turns):
        if turn.probability >= HIGH:
            qs.append(0.0 if i in asked else q[i])
    if not qs:
        return 0.0
    return poisson_binomial_tail(np.array(qs), label_threshold(len(qs), FRAC))


def evpi_order(traj, q, b) -> list[int]:
    """Greedy sequence of up to b questions maximizing the drop in P(event)."""
    asked: list[int] = []
    cands = confident(traj)
    for _ in range(min(b, len(cands))):
        base = event_prob(traj, q, set(asked))
        best, best_gain = None, -1.0
        for i in cands:
            if i in asked:
                continue
            gain = base - event_prob(traj, q, set(asked) | {i})
            if gain > best_gain:
                best, best_gain = i, gain
        asked.append(best)
    return asked


def evpi_gains(traj, q, kmax) -> list[float]:
    """Marginal expected reductions of the greedy EVPI sequence."""
    order = evpi_order(traj, q, kmax)
    gains, prev = [], event_prob(traj, q, set())
    for j in range(len(order)):
        cur = event_prob(traj, q, set(order[: j + 1]))
        gains.append(prev - cur)
        prev = cur
    return gains, order


# ------------------------------------------------------------ main


def run(corpus: str) -> str:
    D = load_with_truth(corpus, N_CORPUS[corpus], 42)
    trajs = [t for t, _ in D]
    wrongs = [w for _, w in D]
    labels = np.array([t.eval_label for t in trajs])
    rng = np.random.default_rng(0)

    rows: dict[str, list[tuple[float, float, float]]] = {}  # policy -> [(qps, hall rate, residual errs)]

    per_item: dict[str, list[np.ndarray]] = {}  # policy -> per-seed arrays (label, residual)

    def record(name, qps, rate, res):
        rows.setdefault(name, []).append((qps, rate, res))

    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids([t.task_id for t in trajs], 0.7, seed)
        tr = [i for i, t in enumerate(trajs) if t.task_id in tr_ids]
        te = [i for i, t in enumerate(trajs) if t.task_id in te_ids]
        det = IndependentPosterior(by_value=True).fit([trajs[i] for i in tr], None, labels[tr])
        Q = {i: turn_posteriors(det, trajs[i]) for i in te}

        def evaluate(name, choose):
            asked_counts, labs, res = [], [], []
            for i in te:
                asked = {int(j) for j in choose(i)}
                asked_counts.append(len(asked))
                lab, r = residual(trajs[i], wrongs[i], asked)
                labs.append(lab)
                res.append(r)
            per_item.setdefault(name, []).append(np.array([labs, res], dtype=float))
            record(name, float(np.mean(asked_counts)), float(np.mean(labs)), float(np.mean(res)))

        evaluate("none", lambda i: [])
        evaluate("detailed_questions",
                 lambda i: [j for j, t in enumerate(trajs[i].turns) if t.attribute in UNDERDETERMINED])
        evaluate("detailed_all", lambda i: range(len(trajs[i].turns)))
        def pick(i, keyfn, b):
            c = confident(trajs[i])
            return sorted(c, key=keyfn)[:b]

        for b in BUDGETS:
            evaluate(f"random@{b}", lambda i, b=b: list(rng.permutation(confident(trajs[i])))[:b])
            evaluate(f"lowconf@{b}", lambda i, b=b: pick(i, lambda j, i=i: trajs[i].turns[j].probability, b))
            evaluate(f"posterior@{b}", lambda i, b=b: pick(i, lambda j, i=i: -Q[i][j], b))
            evaluate(f"evpi@{b}", lambda i, b=b: evpi_order(trajs[i], Q[i], b))
            evaluate(f"oracle@{b}", lambda i, b=b: [j for j, t in enumerate(trajs[i].turns)
                                                    if t.probability >= HIGH and wrongs[i][j]][:b])

        # Global budget: spend B * n_test questions on the largest marginal
        # expected reductions across all test sessions.
        kmax = max(BUDGETS) + 3
        cand = []  # (gain, session, rank)
        orders = {}
        for i in te:
            g, order = evpi_gains(trajs[i], Q[i], kmax)
            orders[i] = order
            cand.extend((gain, i, r) for r, gain in enumerate(g))
        cand.sort(key=lambda x: -x[0])
        for b in (0.25, 0.5, 1, 2, 3):
            total = int(round(b * len(te)))
            take = {}
            # greedy marginal gains are non-increasing within a session, so
            # taking the top-`total` entries respects per-session order
            for gain, i, r in cand[:total]:
                take[i] = max(take.get(i, 0), r + 1)
            evaluate(f"evpi_global@{b}", lambda i, take=take: orders[i][: take.get(i, 0)])

    lines = [f"# e8_clarification — {corpus}, test splits seeds {SEEDS}, no-cascade, no-dilution", "",
             f"{'policy':22s} {'questions/session':>18s} {'hallucination rate':>19s} {'residual conf. errors':>22s}   per-seed rate"]
    order = sorted(rows, key=lambda n: (np.mean([r[0] for r in rows[n]]), -np.mean([r[1] for r in rows[n]])))
    for name in order:
        qps = np.mean([r[0] for r in rows[name]])
        rate = np.mean([r[1] for r in rows[name]])
        res = np.mean([r[2] for r in rows[name]])
        lines.append(f"{name:22s} {qps:18.2f} {rate:19.3f} {res:22.3f}   " + " ".join(f"{r[1]:.3f}" for r in rows[name]))
    # Paired bootstrap over sessions (test sets of the 3 seeds concatenated,
    # same practice as e6) for the key comparisons; negative = first is better.
    comps = [("posterior@1", "detailed_questions"), ("evpi_global@1", "detailed_questions"),
             ("posterior@2", "detailed_questions"), ("posterior@1", "lowconf@1"),
             ("posterior@1", "random@1"), ("posterior@2", "lowconf@2")]
    brng = np.random.default_rng(1)
    lines += ["", "## paired bootstrap (2000 resamples): A - B; negative = A better",
              f"{'A':16s} {'B':20s} {'d hallucination rate':>32s} {'d residual errors':>30s}"]
    for a, b in comps:
        A = np.concatenate(per_item[a], axis=1)
        B = np.concatenate(per_item[b], axis=1)
        d = A - B
        n = d.shape[1]
        boots = np.array([d[:, brng.integers(0, n, n)].mean(axis=1) for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        m = d.mean(axis=1)
        lines.append(f"{a:16s} {b:20s} {m[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]".ljust(70)
                     + f"{m[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]")
    out = REPO / "results" / "e8_clarification" / f"{corpus}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="70B_42", choices=sorted(N_CORPUS))
    args = ap.parse_args()
    print(run(args.corpus))


if __name__ == "__main__":
    main()
