"""e12: EXPLORATORY analyses requested in review (not pre-registered).

    python3 -m eval.e12_exploratory

Two questions the confirmatory plan does not ask, run on the same confirmatory
corpora, splits and bootstrap as eval/e11_confirmatory.py:

1. Intake-aware supervised detectors (deployable). The rules-aware analysis
   (Amendment 2) shows that the population risk given the intake is high,
   but no registered deployable detector sees the intake answers. intake_lr
   is ctx_lr plus the five intake answers (one-hot) and every
   (attribute, value, intake question, intake answer) cell of the confident
   commitments; intake_lr+fb adds ctx_lr+fb's reaction cells. Same L2
   logistic regression, C = 0.1, no tuning.
2. Per-commitment risk (the posterior's q_t). On the test split, every
   confident commitment on a user-dependent attribute: AUROC of q_t for
   that commitment's error, and expected calibration error (10 equal-width
   bins). V6 is fit on train exactly as in e11 (label L_U).

Writes results/e12_exploratory/{summary.md, numbers.json}. Every number from
here is labelled exploratory wherever it is reported.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e11_confirmatory import CONFIRM, MIN_SESSIONS, REPO, TRAVEL, build_dag, load, mean_auc, strat_boot
from eval.e7_common import SEEDS, TRAIN_FRAC, run_protocol
from eval.score_trajectories import split_task_ids
from pge.baselines.context_lr import ContextLR
from pge.baselines.posterior import ChannelPosterior

OUT = REPO / "results" / "e12_exploratory"


class IntakeLR(ContextLR):
    """ctx_lr(+fb) plus the intake answers and commitment x intake cells."""
    SETUP: dict = {}  # task_id -> setup answers (detector-visible: the agent's prompt contains them)

    def __init__(self, with_feedback=False):
        super().__init__(with_feedback=with_feedback)
        self.name = "intake_lr" + ("+fb" if with_feedback else "")

    def _traj_keys(self, traj):
        setup = self.SETUP[traj.task_id]
        for q, a in setup.items():
            yield ("i", q, str(a))
        for turn in traj.turns:
            if turn.probability < self.high:
                continue
            yield from self._keys(turn)
            for q, a in setup.items():
                yield ("ci", turn.attribute, turn.guess_value, q, str(a))

    def _features(self, traj):
        x = np.zeros(len(self.vocab))
        for k in self._traj_keys(traj):
            if k in self.vocab:
                x[self.vocab[k]] += 1
        h = sum(t.probability >= self.high for t in traj.turns)
        return np.concatenate([x, [h, len(traj.turns), float(np.mean([t.probability for t in traj.turns]))]])

    def fit(self, trajectories, dags=None, labels=None):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(labels, dtype=int)
        keys = {k for t in trajectories for k in self._traj_keys(t)}
        self.vocab = {k: i for i, k in enumerate(sorted(keys, key=str))}
        X = np.stack([self._features(t) for t in trajectories])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        if len(set(y.tolist())) < 2:
            self.const = float(y.mean())
            return self
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mu) / self.sd, y)
        return self


def ece(q, w, bins=10):
    q, w = np.asarray(q), np.asarray(w)
    idx = np.minimum((q * bins).astype(int), bins - 1)
    return float(sum(abs(q[idx == b].mean() - w[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def per_commitment(trajs, LU, sessions, dom):
    ids = [t.task_id for t in trajs]
    qs, ws = [], []
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids(ids, TRAIN_FRAC, seed)
        tr = [t for t in trajs if t.task_id in tr_ids]
        ytr = [LU[i] for i, t in enumerate(trajs) if t.task_id in tr_ids]
        det = ChannelPosterior(event_attrs=dom.USER_DEPENDENT_SCHEMA)
        det.fit(tr, [build_dag(t) for t in tr], ytr)
        for t in trajs:
            if t.task_id not in te_ids:
                continue
            s = sessions[t.task_id]
            U = dom.user_dependent_attrs(s.setup_answers)
            q = det.turn_q(t)
            for turn, qt in zip(t.turns, q):
                if turn.probability >= det.high and turn.attribute in U:
                    qs.append(float(qt))
                    ws.append(int(turn.guess_value != s.user_state[turn.attribute]))
    qs, ws = np.array(qs), np.array(ws)
    return {"n": int(len(ws)) // len(SEEDS), "wrong": float(ws.mean()), "mean_q": float(qs.mean()),
            "auroc_q": float(auroc(qs, ws)), "ece_q": ece(qs, ws)}


def block(corpora, title):
    lines, numbers, runs = [f"## {title}", ""], {}, {"L_U": {}, "L": {}}
    for tag, path, seed, n in corpora:
        if not path.exists():
            lines.append(f"> {tag}: not found; skipped.")
            continue
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        dags = [build_dag(t) for t in trajs]
        F = [("ctx_lr", lambda: ContextLR()), ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
             ("intake_lr", lambda: IntakeLR()), ("intake_lr+fb", lambda: IntakeLR(True)),
             ("V6", lambda: ChannelPosterior(event_attrs=dom.USER_DEPENDENT_SCHEMA))]
        for lab, y in (("L_U", LU), ("L", L)):
            Fl = F if lab == "L_U" else F[:4] + [("V6", lambda: ChannelPosterior())]
            R = run_protocol(trajs, y, Fl, dags=dags)
            runs[lab][tag] = (R, len(trajs))
            numbers.setdefault(lab, {})[tag] = {d: mean_auc(R, d) for d in R["auroc"]}
        pc = per_commitment(trajs, LU, sessions, dom)
        numbers.setdefault("percommit_q", {})[tag] = pc
        print(f"[{tag}] " + " ".join(f"{d} {v:.3f}" for d, v in numbers["L_U"][tag].items()) +
              f" | q_t AUROC {pc['auroc_q']:.3f} ECE {pc['ece_q']:.3f}", flush=True)
    for lab in ("L_U", "L"):
        if not runs[lab]:
            continue
        lines += [f"### {lab}: session AUROC (mean over split seeds)", "",
                  "| corpus | ctx_lr | ctx_lr+fb | intake_lr | intake_lr+fb | V6 |", "|---|---|---|---|---|---|"]
        for tag, d in numbers[lab].items():
            lines.append(f"| {tag} | " + " | ".join(f"{d[k]:.3f}" for k in
                                                   ("ctx_lr", "ctx_lr+fb", "intake_lr", "intake_lr+fb", "V6")) + " |")
        pool = [R for R, n in runs[lab].values() if n >= MIN_SESSIONS]
        if len(pool) >= 2:
            lines += ["", f"Pooled over {len(pool)} corpora (stratified cluster bootstrap):", ""]
            for a, b in (("intake_lr", "ctx_lr"), ("intake_lr+fb", "intake_lr"), ("intake_lr+fb", "ctx_lr+fb"),
                         ("V6", "intake_lr+fb"), ("intake_lr", None), ("intake_lr+fb", None)):
                ci = strat_boot(pool, a, b)
                q = f"AUROC({a})" if b is None else f"{a} - {b}"
                numbers[lab].setdefault("pooled", {})[q] = ci
                lines.append(f"- {q}: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        lines.append("")
    if numbers.get("percommit_q"):
        lines += ["### Per-commitment q_t (V6, L_U fit; confident user-dependent commitments, test splits)", "",
                  "| corpus | n per split | wrong | mean q | AUROC(q) | ECE(q) |", "|---|---|---|---|---|---|"]
        for tag, d in numbers["percommit_q"].items():
            lines.append(f"| {tag} | {d['n']} | {d['wrong']:.3f} | {d['mean_q']:.3f} | {d['auroc_q']:.3f} | {d['ece_q']:.3f} |")
    return lines, numbers


def main():
    t0 = time.time()
    head = ["# e12 EXPLORATORY (not pre-registered; requested in review)", "",
            "Same corpora, splits (42/123/456, 70/30) and bootstrap as e11.", ""]
    lines, numbers = block(CONFIRM, "Healthcare, confirmatory corpora")
    tl, tn = block(TRAVEL, "Travel (second domain)")
    if any("|" in x for x in tl):
        lines += [""] + tl
        numbers["travel_v1"] = tn
    lines.append(f"\nruntime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(head + lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(head + lines))


if __name__ == "__main__":
    main()
