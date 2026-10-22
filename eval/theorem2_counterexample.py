"""Counterexample to Theorem 2's monotonicity when the agent conditions on the reactions.

Theorem 2 (less noisy feedback cannot hurt) fixes the joint law of (u, X, C)
and regenerates the reactions for the detector only. This script shows the
premise is essential: if the agent itself conditions on the same reactions,
the law of (X, C) changes with the flip rate eps and the Bayes detector's
AUROC can be strictly HIGHER at eps = 1/2 (useless feedback) than at
eps = 0 (perfect feedback).

Construction (two turns, exact enumeration, no sampling):
  - Private bit y with P(y=1) = 0.9. Turn 1 the agent probes attribute A with
    the non-confident commitment v1 = 0 (usually wrong) and receives the
    reaction F1 = (correctness sign of v1) flipped with probability eps.
  - Policy: if accepted (F1 = +1) the agent confidently recommits v2 = 0 on A,
    which elicits a reaction F2 at the same eps. If contradicted (F1 = -1) it
    abandons A and confidently commits value 0 on an opaque attribute B whose
    truth z ~ Bernoulli(1/2) is independent of y; B never elicits reactions.
  - The session label is L = w2 (the only confident commitment).

Mechanism: reliable contradictions route exactly the sessions where the agent
is wrong about y into the opaque branch, where its error is a coin flip and
carries no evidence; at eps = 1/2 the routing is uninformative, errors stay on
the reactive attribute, and the branch identity alone separates the label
better. The detector scored here is eta_eps = P(L=1 | branch, F1, F2), the
Neyman-Pearson-optimal detector of L from the eps-feedback, so the reversal
is not an artifact of a weak detector.

Offline, exact, $0. Artifact: results/theorem2_counterexample/summary.txt
"""

from fractions import Fraction
from pathlib import Path


P_Y1 = Fraction(9, 10)  # P(y = 1); the agent's probe v1 = 0 is usually wrong


def outcome_law(eps):
    """Joint law of (observable, L) as {(obs, L): prob}, exact in Fractions.

    obs is ("A", F1, F2) on the reactive branch or ("B", F1) on the opaque one.
    """
    eps = Fraction(eps) if not isinstance(eps, Fraction) else eps
    law = {}
    for y, py in ((1, P_Y1), (0, 1 - P_Y1)):
        w1 = 1 if y != 0 else 0  # v1 = 0
        for f1 in (+1, -1):
            correct_sign = -1 if w1 else +1
            pf1 = (1 - eps) if f1 == correct_sign else eps
            if pf1 == 0:
                continue
            if f1 == +1:
                # recommit v2 = 0 on A; w2 = 1[y != 0]; F2 observed
                w2 = w1
                for f2 in (+1, -1):
                    c2 = -1 if w2 else +1
                    pf2 = (1 - eps) if f2 == c2 else eps
                    if pf2 == 0:
                        continue
                    k = (("A", f1, f2), w2)
                    law[k] = law.get(k, Fraction(0)) + py * pf1 * pf2
            else:
                # opaque attribute B: truth z ~ Bern(1/2), no reaction
                for w2, pz in ((0, Fraction(1, 2)), (1, Fraction(1, 2))):
                    k = (("B", f1), w2)
                    law[k] = law.get(k, Fraction(0)) + py * pf1 * pz
    assert sum(law.values()) == 1
    return law


def bayes_auroc(law):
    """AUROC of eta(obs) = P(L=1 | obs), exact, ties counted 1/2."""
    obs_mass = {}
    for (obs, lab), p in law.items():
        m = obs_mass.setdefault(obs, [Fraction(0), Fraction(0)])
        m[lab] += p
    p1 = sum(m[1] for m in obs_mass.values())
    p0 = 1 - p1
    if p1 == 0 or p0 == 0:
        return None
    # score of obs = posterior; positive/negative score distributions
    pts = []
    for obs, (m0, m1) in obs_mass.items():
        eta = m1 / (m0 + m1)
        pts.append((eta, m1 / p1, m0 / p0))
    auc = Fraction(0)
    for eta_p, a, _ in pts:
        for eta_n, _, b in pts:
            if eta_p > eta_n:
                auc += a * b
            elif eta_p == eta_n:
                auc += a * b / 2
    return auc


def main():
    grid = [Fraction(k, 20) for k in range(0, 11)]  # 0, 0.05, ..., 0.5
    rows = []
    for eps in grid:
        auc = bayes_auroc(outcome_law(eps))
        rows.append((eps, auc))

    lines = [
        "Theorem 2 counterexample: agent conditions on the same reactions",
        f"P(y=1) = {P_Y1}; policy: contradicted -> opaque attribute B (coin-flip",
        "truth, no reactions); accepted -> recommit on A with a reaction.",
        "Label L = w2. Score = Bayes posterior P(L=1 | branch, F1, F2).",
        "",
        "  eps    AUROC(eta_eps)",
    ]
    for eps, auc in rows:
        lines.append(f"  {float(eps):.2f}   {float(auc):.4f}")

    a0 = dict(rows)[Fraction(0)]
    ah = dict(rows)[Fraction(1, 2)]
    lines += [
        "",
        f"AUROC at eps = 0   (perfect feedback): {float(a0):.4f}",
        f"AUROC at eps = 1/2 (useless feedback): {float(ah):.4f}",
        "Monotonicity of Theorem 2 FAILS here: the premise (trajectories fixed,",
        "reactions regenerated for the detector only) is essential.",
    ]
    out = "\n".join(lines)
    print(out)

    # the actual counterexample, verified exactly
    assert a0 < ah, "expected AUROC(eps=0) < AUROC(eps=1/2)"

    dst = Path(__file__).resolve().parent.parent / "results" / "theorem2_counterexample"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.txt").write_text(out + "\n")
    print(f"\nwritten: {dst / 'summary.txt'}")


if __name__ == "__main__":
    main()
