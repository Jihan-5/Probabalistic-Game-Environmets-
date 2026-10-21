"""e17: stronger supervised comparators for V7, and V7 without reactions (exploratory).

    python3 -m eval.e17_strong_baselines

Review asked whether "the best supervised detector given the same information"
is a fair comparator. Every detector below sees exactly V7's inputs (intake
answers, commitments, reactions, the schema's dependency lists) and the same
training labels; none is tuned on test sessions:

  intake_lr+fb       the H8 comparator (C = 0.1 fixed), for reference
  intake_lr+fb (cv)  same features, C chosen by 5-fold CV on the training split
  dep_lr+fb (cv)     V7's own structure as features: counts per (attribute,
                     value, intake answers it depends on) cell and per
                     (cell, reaction), user-dependent-schema attributes only;
                     C by 5-fold CV on the training split
  intake_gbm+fb      gradient-boosted trees on intake_lr+fb's features
                     (fixed: depth 3, rate 0.05, 300 iterations, early stopping
                     on a 20% slice of the training split)
  V7 / V7-prior      the frozen V7, and V7 with every reaction's likelihood
                     ratio set to 1 (same fitted priors and event): what the
                     user's reactions add to V7.

Same splits and bootstrap as e11/e16; healthcare confirmatory corpora present
(L_U). Writes results/e17_strong_baselines/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from eval.e11_confirmatory import CONFIRM, MIN_SESSIONS, REPO, build_dag, load, mean_auc, pooled_rows, strat_boot
from eval.e12_exploratory import IntakeLR
from eval.e7_common import cluster_boot_diff, run_protocol
from pge.baselines.intake_posterior import IntakePosterior, intake_keys, v7
from pge.baselines.posterior import poisson_binomial_tail, label_threshold, turn_signal

OUT = REPO / "results" / "e17_strong_baselines"
CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


class IntakeLRCV(IntakeLR):
    def __init__(self):
        super().__init__(with_feedback=True)
        self.name = "intake_lr+fb (cv)"

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels, dtype=int)
        keys = {k for t in trajectories for k in self._traj_keys(t)}
        self.vocab = {k: i for i, k in enumerate(sorted(keys, key=str))}
        X = np.stack([self._features(t) for t in trajectories])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.model = LogisticRegressionCV(Cs=list(CS), cv=5, scoring="roc_auc", max_iter=5000).fit(
            (X - self.mu) / self.sd, y)
        return self


class IntakeGBM(IntakeLR):
    def __init__(self):
        super().__init__(with_feedback=True)
        self.name = "intake_gbm+fb"

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels, dtype=int)
        keys = {k for t in trajectories for k in self._traj_keys(t)}
        self.vocab = {k: i for i, k in enumerate(sorted(keys, key=str))}
        X = np.stack([self._features(t) for t in trajectories])
        self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        self.model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                    early_stopping=True, validation_fraction=0.2,
                                                    random_state=0).fit(X, y)
        return self


class DepLRCV:
    """V7's structure as LR features: (attribute, value, intake-dependency cell) and
    (cell, reaction) counts over confident commitments on schema user-dependent attributes."""
    SETUP: dict = {}

    def __init__(self, dom):
        self.name = "dep_lr+fb (cv)"
        self.K = intake_keys(dom.ATTRIBUTES, dom.SETUP_QUESTIONS)
        self.S = dom.USER_DEPENDENT_SCHEMA

    def _keys(self, traj):
        setup = self.SETUP[traj.task_id]
        for turn in traj.turns:
            if turn.probability < 0.6 or turn.attribute not in self.S:
                continue
            c = tuple((k, str(setup[k])) for k in self.K.get(turn.attribute, ()))
            yield ("c", turn.attribute, turn.guess_value, c)
            yield ("f", turn.attribute, turn.guess_value, c, turn_signal(turn))

    def _x(self, traj):
        x = np.zeros(len(self.vocab) + 1)
        for k in self._keys(traj):
            if k in self.vocab:
                x[self.vocab[k]] += 1
        x[-1] = sum(t.probability >= 0.6 and t.attribute in self.S for t in traj.turns)
        return x

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels, dtype=int)
        self.vocab = {k: i for i, k in enumerate(sorted({k for t in trajectories for k in self._keys(t)}, key=str))}
        X = np.stack([self._x(t) for t in trajectories])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.model = LogisticRegressionCV(Cs=list(CS), cv=5, scoring="roc_auc", max_iter=5000).fit(
            (X - self.mu) / self.sd, y)
        return self

    def score(self, trajectory, dag=None):
        return float(self.model.predict_proba(((self._x(trajectory) - self.mu) / self.sd)[None, :])[0, 1])


class V7Prior:
    """The frozen V7 fitted as usual, scored with every reaction ignored."""

    def __init__(self, dom):
        self.det = v7(dom, "L_U")
        self.name = "V7-prior"

    def fit(self, trajectories, dags=None, labels=None):
        self.det.fit(trajectories, dags, labels)
        return self

    def score(self, trajectory, dag=None):
        d = self.det
        d._setup = d.INTAKE[trajectory.task_id]
        q = [d._prior(t) for t in trajectory.turns if d._in_event_ctx(t)]
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), d.frac)) if q else 0.0


def main():
    t0 = time.time()
    lines = ["# e17: stronger supervised comparators and V7 without reactions (EXPLORATORY; L_U)", ""]
    numbers, runs = {}, {}
    for tag, path, seed, n in CONFIRM:
        if not path.exists():
            lines.append(f"> {tag}: not found; skipped.")
            continue
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakeLR.SETUP = IntakePosterior.INTAKE = DepLRCV.SETUP = {
            t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        F = [("V7", lambda: v7(dom, "L_U")), ("V7-prior", lambda: V7Prior(dom)),
             ("intake_lr+fb", lambda: IntakeLR(True)), ("intake_lr+fb (cv)", IntakeLRCV),
             ("dep_lr+fb (cv)", lambda: DepLRCV(dom)), ("intake_gbm+fb", IntakeGBM)]
        R = run_protocol(trajs, LU, F, dags=[build_dag(t) for t in trajs])
        runs[tag] = (R, len(trajs))
        yy, g = np.concatenate(R["y"]), np.concatenate(R["tid"])
        numbers[tag] = {"auroc": {d: mean_auc(R, d) for d in R["auroc"]},
                        "diffs": {f"V7 - {b}": cluster_boot_diff(pooled_rows(R, "V7"), pooled_rows(R, b), yy, g, n=2000)
                                  for b in R["auroc"] if b != "V7"}}
        print(tag, {k: round(v, 3) for k, v in numbers[tag]["auroc"].items()}, f"{time.time() - t0:.0f}s", flush=True)
    dets = list(next(iter(numbers.values()))["auroc"])
    lines += ["| corpus | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
    lines += [f"| {t} | " + " | ".join(f"{d['auroc'][k]:.3f}" for k in dets) + " |" for t, d in numbers.items()]
    lines.append("")
    for t, d in numbers.items():
        lines.append(f"- {t}: " + "; ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in d["diffs"].items()))
    pool = [R for R, n in runs.values() if n >= MIN_SESSIONS]
    if len(pool) >= 2:
        lines += ["", f"Pooled over {len(pool)} corpora:", ""]
        numbers["pooled"] = {}
        for b in dets:
            if b == "V7":
                continue
            ci = strat_boot(pool, "V7", b)
            numbers["pooled"][f"V7 - {b}"] = ci
            lines.append(f"- V7 - {b}: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
    lines.append(f"\nruntime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
