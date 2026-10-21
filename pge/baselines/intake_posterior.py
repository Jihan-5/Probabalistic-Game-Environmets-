"""V7: the label-event posterior with intake-conditioned cells (Amendment 6).

The frozen detector is `v7(dom, label)` at the end of this module. For L_U it
uses event="learned": cells (attribute, value, bucket, intake answers the
attribute depends on in the schema) shrunk toward V6's cell, and a label event
learned per (attribute, intake) cell from training labels (min_cell=2, two
rounds of channel/membership fitting). For L it uses the intake cells only.
The prior-threshold event ("delta") is a development variant that was not kept
(results/e13_v7_dev/). This docstring was corrected after the freeze; the code
is unchanged since the Amendment 6 commit.

Motivation (development corpora; results/e13_v7_dev): Proposition 1's
assumption (A1) asks that a commitment's error depend on the history only
through its cell. In the simulator it also depends on the intake answers, and
the user-specific label counts a commitment only when its truth is NOT fixed
by the intake. V6 models neither.

Detector-visible inputs only: the intake answers (the agent's own input; set
per corpus in INTAKE, keyed by task id), the schema's dependency lists and the
schema-level user-dependent set, and training labels. No rules, no hidden
facts, no truth.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pge.baselines.posterior import (
    BUCKETS, R0_GRID, R1_GRID, ChannelPosterior, _bce, conf_bucket, label_threshold,
    poisson_binomial_tail, turn_signal,
)

DELTA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)
MIN_CELL = 10  # sessions a (attribute, intake) cell needs before its membership is learned


def intake_keys(attributes: dict, setup_questions) -> dict:
    """attribute -> sorted intake questions it depends on (transitively)."""
    setup = set(setup_questions)
    memo: dict = {}

    def keys(a):
        if a not in memo:
            memo[a] = set()
            for d in attributes[a][1]:
                if d.startswith("attr:"):
                    memo[a] |= keys(d[5:])
                elif not d.startswith("hidden:") and d in setup:
                    memo[a].add(d)
        return memo[a]

    return {a: tuple(sorted(keys(a))) for a in attributes}


class IntakePosterior(ChannelPosterior):
    INTAKE: dict = {}  # task_id -> intake answers (the agent's input)

    def __init__(self, dom, prior_strength: float = 5.0, event_attrs=None, delta_grid=DELTA_GRID,
                 event: str = "delta", min_cell: int = MIN_CELL, rounds: int = 2, halves: bool = False):
        super().__init__(prior_strength=prior_strength, event_attrs=event_attrs)
        self.name = "V7" if event == "delta" else "V7m"
        self.event = event  # "delta": prior-threshold event; "learned": membership per (attribute, intake) cell
        self.min_cell, self.rounds = min_cell, rounds
        self.halves = halves  # accept a membership change only if it lowers train NLL in both halves
        self.member: dict = {}
        self.K = intake_keys(dom.ATTRIBUTES, dom.SETUP_QUESTIONS)
        self.delta_grid = tuple(delta_grid)
        self.delta = 0.0
        self._setup = None

    # ---- cells
    def _c(self, attribute, setup):
        return tuple((k, str(setup[k])) for k in self.K.get(attribute, ()))

    def _prior(self, turn):
        b = conf_bucket(turn.probability)
        ab = self.pi.get(("ab", turn.attribute, b), self.pi_bucket.get(b, 0.5))
        avb = self.pi.get(("avb", turn.attribute, turn.guess_value, b), ab)
        if self._setup is None:
            return avb
        return self.pi.get(("avbc", turn.attribute, turn.guess_value, b, self._c(turn.attribute, self._setup)), avb)

    def _counts(self, trajs):
        neg, pos, obs = defaultdict(float), defaultdict(float), defaultdict(float)
        for t in trajs:
            setup = self.INTAKE[t.task_id]
            for turn in t.turns:
                if turn.attribute not in self.observable:
                    continue
                s = turn_signal(turn)
                b = conf_bucket(turn.probability)
                a, v = turn.attribute, turn.guess_value
                for key in (("b", b), ("ab", a, b), ("avb", a, v, b), ("avbc", a, v, b, self._c(a, setup))):
                    neg[key] += s < 0
                    pos[key] += s > 0
                    obs[key] += 1
        return neg, pos, obs

    def _channel_from_counts(self, counts, r0, r1):
        neg, pos, obs = counts

        def mom(n, o):
            return float(np.clip((n / o - r0) / (r1 - r0), 0.01, 0.99)) if o else 0.5

        def shrink(key, parent):
            w = obs[key] / (obs[key] + self.k)
            return w * mom(neg[key], obs[key]) + (1 - w) * parent

        pi_bucket = {b: mom(neg[("b", b)], obs[("b", b)]) for b in range(len(BUCKETS) + 1)}
        pi = {k: shrink(k, pi_bucket[k[2]]) for k in obs if k[0] == "ab"}
        for k in obs:
            if k[0] == "avb":
                pi[k] = shrink(k, pi[("ab", k[1], k[3])])
        for k in obs:
            if k[0] == "avbc":
                pi[k] = shrink(k, pi[("avb", k[1], k[2], k[3])])
        # confirmation rates: as V6 (weighted least squares on the avb cells)
        xs, ys, ws = [], [], []
        for k in obs:
            if k[0] == "avb" and obs[k] > 0:
                xs.append(pi[k]); ys.append(pos[k] / obs[k]); ws.append(obs[k])
        xs, ys, ws = (np.array(v, dtype=float) for v in (xs, ys, ws))
        if len(xs) >= 2 and np.ptp(xs) > 1e-6:
            X = np.stack([np.ones_like(xs), xs], axis=1)
            W = np.sqrt(ws)[:, None]
            coef, *_ = np.linalg.lstsq(X * W, ys * W[:, 0], rcond=None)
            g0, g1 = float(coef[0]), float(coef[0] + coef[1])
        else:
            g0 = g1 = float(np.average(ys, weights=ws)) if len(ys) else 0.0
        g0 = float(np.clip(g0, 0.0, 1.0 - r0 - 1e-3))
        g1 = float(np.clip(g1, 0.0, 1.0 - r1 - 1e-3))
        return pi, pi_bucket, (r0, r1), (g0, g1)

    def _fit_channel(self, trajs, r0, r1):
        return self._channel_from_counts(self._counts(trajs), r0, r1)

    # ---- event
    def _in_event_ctx(self, turn):
        if not self._in_event(turn):
            return False
        if self.event == "learned":
            return self.member.get((turn.attribute, self._c(turn.attribute, self._setup)), True)
        if self.delta <= 0:
            return True
        p = self._prior(turn)
        return self.delta <= p <= 1 - self.delta

    def _score(self, traj):
        self._setup = self.INTAKE[traj.task_id]
        q = [self._q_turn(t) for t in traj.turns if self._in_event_ctx(t)]
        if not q:
            return 0.0
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), self.frac))

    def turn_q(self, traj):
        self._setup = self.INTAKE[traj.task_id]
        return np.array([self._q_turn(t) for t in traj.turns])

    def fit(self, trajectories, dags=None, labels=None):
        if self.event == "learned":
            return self._fit_learned(trajectories, labels)
        return self._fit_delta(trajectories, labels)

    # ---- learned label event: which (attribute, intake) cells count as user-specific
    def _prep(self, trajectories):
        prep = []
        for t in trajectories:
            setup = self.INTAKE[t.task_id]
            rows = []
            for turn in t.turns:
                if not self._in_event(turn):
                    continue
                b = conf_bucket(turn.probability)
                a, v = turn.attribute, turn.guess_value
                c = self._c(a, setup)
                rows.append((("avbc", a, v, b, c), ("avb", a, v, b), ("ab", a, b), b,
                             turn_signal(turn), a in self.observable, (a, c)))
            prep.append(rows)
        return prep

    @staticmethod
    def _qs(prep, ch):
        pi, pib, r, g = ch
        miss = max(1.0 - r[1] - g[1], 1e-3), max(1.0 - r[0] - g[0], 1e-3)
        out = []
        for rows in prep:
            qq = []
            for kc, kv, ka, b, sgn, ob, cell in rows:
                pr = pi.get(kc, pi.get(kv, pi.get(ka, pib.get(b, 0.5))))
                if not ob:
                    l1 = l0 = 1.0
                elif sgn < 0:
                    l1, l0 = r[1], r[0]
                elif sgn > 0:
                    l1, l0 = g[1], g[0]
                else:
                    l1, l0 = miss
                qq.append((cell, pr * l1 / (pr * l1 + (1 - pr) * l0)))
            out.append(qq)
        return out

    def _tail(self, qq, member):
        q = [x for cell, x in qq if member.get(cell, True)]
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), self.frac)) if q else 0.0

    @staticmethod
    def _nll(p, y):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _fit_learned(self, trajectories, labels):
        y = np.asarray(labels, dtype=float)
        self.observable = {turn.attribute for t in trajectories for turn in t.turns if turn_signal(turn) != 0}
        counts = self._counts(trajectories)
        prep = self._prep(trajectories)
        grid = [(r0, r1) for r0 in R0_GRID for r1 in R1_GRID if r1 > r0 + 0.1 - 1e-9]
        member: dict = {}
        ch = None
        for _ in range(self.rounds):
            best = None
            for r0, r1 in grid:  # channel given the current membership
                c = self._channel_from_counts(counts, r0, r1)
                qs = self._qs(prep, c)
                loss = sum(self._nll(self._tail(qq, member), yi) for qq, yi in zip(qs, y))
                if best is None or loss < best[0]:
                    best = (loss, c, qs)
            _, ch, qs = best
            # greedy membership on train NLL, cells with enough sessions, most frequent first
            sess_of = defaultdict(set)
            for i, qq in enumerate(qs):
                for cell, _q in qq:
                    sess_of[cell].add(i)
            cells = sorted((c for c in sess_of if len(sess_of[c]) >= self.min_cell),
                           key=lambda c: (-len(sess_of[c]), str(c)))
            cur = np.array([self._nll(self._tail(qq, member), yi) for qq, yi in zip(qs, y)])
            half = np.random.default_rng(0).permutation(len(y)) % 2
            for _pass in range(4):
                changed = False
                for cell in cells:
                    idx = sorted(sess_of[cell])
                    trial = dict(member)
                    trial[cell] = not member.get(cell, True)
                    new = np.array([self._nll(self._tail(qs[i], trial), y[i]) for i in idx])
                    d = new - cur[idx]
                    h = half[idx]
                    ok = (d[h == 0].sum() < 0 and d[h == 1].sum() < 0) if self.halves else d.sum() < -1e-6
                    if ok:
                        member, cur[idx], changed = trial, new, True
                if not changed:
                    break
        self.pi, self.pi_bucket, self.r, self.g = ch
        self.member = member
        self.eps = None
        return self

    def _fit_delta(self, trajectories, labels):
        """Same objective as scoring every grid point with _score (train-label
        BCE over (r0, r1, delta)); counts and per-turn keys are computed once."""
        y = np.asarray(labels, dtype=float)
        self.observable = {turn.attribute for t in trajectories for turn in t.turns if turn_signal(turn) != 0}
        counts = self._counts(trajectories)
        prep = []
        for t in trajectories:
            setup = self.INTAKE[t.task_id]
            rows = []
            for turn in t.turns:
                if not self._in_event(turn):
                    continue
                b = conf_bucket(turn.probability)
                a, v = turn.attribute, turn.guess_value
                rows.append((("avbc", a, v, b, self._c(a, setup)), ("avb", a, v, b), ("ab", a, b), b,
                             turn_signal(turn), a in self.observable))
            prep.append(rows)
        best = None
        for r0 in R0_GRID:
            for r1 in R1_GRID:
                if r1 <= r0 + 0.1 - 1e-9:
                    continue
                pi, pib, r, g = self._channel_from_counts(counts, r0, r1)
                miss = max(1.0 - r[1] - g[1], 1e-3), max(1.0 - r[0] - g[0], 1e-3)
                sess = []
                for rows in prep:
                    pq = []
                    for kc, kv, ka, b, sgn, ob in rows:
                        pr = pi.get(kc, pi.get(kv, pi.get(ka, pib.get(b, 0.5))))
                        if not ob:
                            l1 = l0 = 1.0
                        elif sgn < 0:
                            l1, l0 = r[1], r[0]
                        elif sgn > 0:
                            l1, l0 = g[1], g[0]
                        else:
                            l1, l0 = miss
                        pq.append((pr, pr * l1 / (pr * l1 + (1 - pr) * l0)))
                    sess.append(pq)
                for d in self.delta_grid:
                    sc = []
                    for pq in sess:
                        q = [qq for pr, qq in pq if d <= 0 or d <= pr <= 1 - d]
                        sc.append(poisson_binomial_tail(np.array(q), label_threshold(len(q), self.frac)) if q else 0.0)
                    loss = _bce(np.array(sc), y)
                    if best is None or loss < best[0]:
                        best = (loss, pi, pib, r, g, d)
        _, self.pi, self.pi_bucket, self.r, self.g, self.delta = best
        self.eps = None
        return self


def v7(dom, label: str) -> IntakePosterior:
    """The frozen V7 (Amendment 6), chosen on the development corpora only
    (results/e13_v7_dev/{summary,sweep,sweep2}.md):
      L_U: intake-conditioned cells + label event learned per (attribute, intake)
           cell from training labels, cells with >= 2 training sessions,
           2 rounds of channel / membership fitting;
      L:   intake-conditioned cells only (L's event is every confident
           commitment by definition)."""
    if label == "L_U":
        return IntakePosterior(dom, event_attrs=dom.USER_DEPENDENT_SCHEMA, event="learned", min_cell=2, rounds=2)
    return IntakePosterior(dom, event_attrs=None, delta_grid=(0.0,))
