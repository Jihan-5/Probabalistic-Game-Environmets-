"""PGE posterior detectors: the probability of the label-defining event.

The PGE label is an event over latent per-commitment correctness. On
healthcare-v2 it is ">= half of the commitments reported at p >= 0.6 are
wrong" (eval_label_v2); on dbbench-v2 it is "the final commitment is
wrong". A detector observes each commitment's reported confidence, the
committed value, the behavioral signal channel and (where cached) k
resamples. The Bayes-optimal score is P(label event | that evidence).

Detectors here (all fit on the train split only, never read eval_label at
score time):

  IndependentPosterior   Healthcare event. Latent errors independent given
                         a prior pi(attribute[, value], confidence bucket).
                         Feedback is the protocol's flip channel: the
                         correctness bit flipped w.p. eps, 0 = unobserved.
                         Priors come from feedback frequencies by the
                         method of moments (no labels); eps is chosen on
                         train by session-label log-likelihood. Exact event
                         probability via the Poisson-binomial tail.
                         by_value=True conditions the prior on the committed
                         value with hierarchical shrinkage (value -> attribute
                         -> bucket). use_samples=True adds a resample-
                         agreement sensor whose emission table is learned by
                         EM, anchored by the feedback channel (no labels).
  DagPosterior           Same event, latent errors follow the dependency DAG
                         (error cascades), exact inference by enumerating
                         all 2^n error configurations. Logged as a negative
                         result: it does not beat IndependentPosterior.
  EvidenceNB             dbbench event (final commitment wrong). Probes have
                         unknown reliability, so emissions are learned from
                         train labels: naive Bayes over per-probe signals,
                         the final turn's resample agreement count and its
                         confidence decile. Laplace alpha=1, untuned.

Results and every variant tried: results/RESULTS.md (2026-10-19 rows) and
results/e6_posterior/.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from pge.baselines.base import TrajectoryScorer
from pge.benchmarks.dbbench_v2 import _extract_final_value

HIGH = 0.6  # healthcare label threshold on reported confidence (detector-visible)
BUCKETS = (0.6, 0.65, 0.75)  # confidence bucket edges for p >= HIGH
EPS_GRID = tuple(np.round(np.arange(0.05, 0.46, 0.025), 3))


def conf_bucket(p: float) -> int:
    """0 for p < 0.6, then 1..3 for the high-confidence bands."""
    if p < BUCKETS[0]:
        return 0
    b = 0
    for i, edge in enumerate(BUCKETS):
        if p >= edge:
            b = i + 1
    return b


def turn_signal(turn) -> int:
    """Collapse named signals into one {-1, 0, +1} (same rule as confidence.py)."""
    vals = list(turn.signals.values())
    return int(np.sign(sum(vals))) if vals else 0


def emission(s: int, w: int, eps: float) -> float:
    """P(observed signal s | latent wrong = w) under the eps-flip channel."""
    if s == 0:
        return 1.0
    if w == 1:
        return 1.0 - eps if s < 0 else eps
    return eps if s < 0 else 1.0 - eps


def poisson_binomial_tail(q: np.ndarray, k: int) -> float:
    """P(sum of independent Bernoulli(q_i) >= k), by dynamic programming."""
    dist = np.zeros(len(q) + 1)
    dist[0] = 1.0
    for qi in q:
        dist[1:] = dist[1:] * (1 - qi) + dist[:-1] * qi
        dist[0] *= 1 - qi
    return float(dist[k:].sum())


def label_threshold(n_high: int, frac: float = 0.5) -> int:
    """Smallest wrong-count with wrong / n_high >= frac (eval_label_v2 rule)."""
    return int(np.ceil(frac * n_high - 1e-12))


def agreement_count(turn):
    """Number of cached resamples whose committed value equals the guess."""
    if not turn.samples:
        return None
    g = str(turn.guess_value).strip().lower()
    return sum(_extract_final_value(s).strip().lower() == g for s in turn.samples)


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------------------------------------------------------- independent


class IndependentPosterior(TrajectoryScorer):
    def __init__(
        self,
        prior_strength: float = 5.0,
        by_value: bool = False,
        use_samples: bool = False,
        em_iters: int = 20,
        high: float = HIGH,
        frac: float = 0.5,
    ):
        self.high, self.frac = high, frac  # label event definition
        self.k = prior_strength
        self.by_value = by_value or use_samples
        self.use_samples = use_samples
        self.em_iters = em_iters
        self.name = (
            "pge_posterior_value+samples"
            if use_samples
            else "pge_posterior_value"
            if by_value
            else "pge_posterior_indep"
        )
        self.eps = None
        self.pi: dict = {}
        self.pi_bucket: dict = {}
        self.A = None  # A[c, w] = P(agreement count c | latent wrong = w)

    def _fit_pi(self, trajs, eps):
        neg, obs = defaultdict(float), defaultdict(float)
        for t in trajs:
            for turn in t.turns:
                s = turn_signal(turn)
                if s == 0:
                    continue
                b = conf_bucket(turn.probability)
                for key in (
                    ("b", b),
                    ("ab", turn.attribute, b),
                    ("avb", turn.attribute, turn.guess_value, b),
                ):
                    neg[key] += s < 0
                    obs[key] += 1
        denom = max(1 - 2 * eps, 1e-3)

        def mom(n, o):  # P(s=-1) = pi(1-eps) + (1-pi)eps
            return float(np.clip((n / o - eps) / denom, 0.01, 0.99)) if o else 0.5

        def shrink(key, parent):
            w = obs[key] / (obs[key] + self.k)
            return w * mom(neg[key], obs[key]) + (1 - w) * parent

        pi_bucket = {b: mom(neg[("b", b)], obs[("b", b)]) for b in range(len(BUCKETS) + 1)}
        pi = {k: shrink(k, pi_bucket[k[2]]) for k in obs if k[0] == "ab"}
        if self.by_value:
            for k in obs:
                if k[0] == "avb":
                    pi[k] = shrink(k, pi[("ab", k[1], k[3])])
        return pi, pi_bucket

    def _prior(self, turn):
        b = conf_bucket(turn.probability)
        ab = self.pi.get(("ab", turn.attribute, b), self.pi_bucket.get(b, 0.5))
        if self.by_value:
            return self.pi.get(("avb", turn.attribute, turn.guess_value, b), ab)
        return ab

    def _fit_A(self, trajs, eps):
        items = []
        for t in trajs:
            for turn in t.turns:
                c = agreement_count(turn)
                if c is None:
                    continue
                s = turn_signal(turn)
                items.append((self._prior(turn), emission(s, 1, eps), emission(s, 0, eps), c))
        if not items:
            return None
        kmax = max(c for *_, c in items)
        A = np.ones((kmax + 1, 2))
        A[:, 0] = (np.arange(kmax + 1) + 1.0) ** 2  # init: correct -> agreement
        A /= A.sum(axis=0, keepdims=True)
        for _ in range(self.em_iters):
            num = np.full((kmax + 1, 2), 0.5)  # light Laplace smoothing
            for pi, l1, l0, c in items:
                a1 = pi * l1 * A[c, 1]
                a0 = (1 - pi) * l0 * A[c, 0]
                q = a1 / (a1 + a0)
                num[c, 1] += q
                num[c, 0] += 1 - q
            A = num / num.sum(axis=0, keepdims=True)
        return A

    def _score(self, traj):
        q = []
        for turn in traj.turns:
            if turn.probability < self.high:
                continue
            pi = self._prior(turn)
            s = turn_signal(turn)
            l1, l0 = emission(s, 1, self.eps), emission(s, 0, self.eps)
            if self.use_samples and self.A is not None:
                c = agreement_count(turn)
                if c is not None and c < len(self.A):
                    l1 *= self.A[c, 1]
                    l0 *= self.A[c, 0]
            q.append(pi * l1 / (pi * l1 + (1 - pi) * l0))
        if not q:
            return 0.0
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), self.frac))

    def fit(self, trajectories, dags=None, labels=None):
        if labels is None:
            raise ValueError(f"{self.name} selects eps on train labels; pass labels")
        y = np.asarray(labels, dtype=float)
        best = None
        for eps in EPS_GRID:
            self.eps = eps
            self.pi, self.pi_bucket = self._fit_pi(trajectories, eps)
            self.A = self._fit_A(trajectories, eps) if self.use_samples else None
            loss = _bce(np.array([self._score(t) for t in trajectories]), y)
            if best is None or loss < best[0]:
                best = (loss, eps, self.pi, self.pi_bucket, self.A)
        _, self.eps, self.pi, self.pi_bucket, self.A = best
        return self

    def score(self, trajectory, dag=None) -> float:
        return self._score(trajectory)


# ---------------------------------------------------------------- DAG model


class DagPosterior(TrajectoryScorer):
    """Latent-error Bayesian network on the dependency DAG (negative result).

    w_t ~ Bernoulli(theta[attr, bucket, any_parent_wrong]); s_t | w_t is the
    eps-flip channel. theta by EM on train signals; eps by train label NLL.
    """

    name = "pge_posterior_dag"

    def __init__(self, parents_fn, prior_strength: float = 5.0, em_iters: int = 10,
                 high: float = HIGH, frac: float = 0.5):
        self.high, self.frac = high, frac
        self.parents_fn = parents_fn  # traj -> list[list[int]]
        self.k = prior_strength
        self.em_iters = em_iters
        self.eps = None
        self.theta: dict = {}
        self.theta_back: dict = {}
        self._cache: dict = {}

    def _config_table(self, n):
        if n not in self._cache:
            self._cache[n] = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int8)
        return self._cache[n]

    def _theta(self, attr, b, pw):
        return self.theta.get((attr, b, pw), self.theta_back.get((b, pw), 0.5))

    def _posterior_configs(self, traj, eps):
        W = self._config_table(len(traj.turns))
        parents = self.parents_fn(traj)
        logp = np.zeros(len(W))
        for i, turn in enumerate(traj.turns):
            b = conf_bucket(turn.probability)
            pw = W[:, parents[i]].max(axis=1) if parents[i] else np.zeros(len(W), dtype=np.int8)
            th = np.where(pw == 1, self._theta(turn.attribute, b, 1), self._theta(turn.attribute, b, 0))
            wi = W[:, i]
            logp += np.log(np.where(wi == 1, th, 1 - th))
            s = turn_signal(turn)
            if s != 0:
                like1 = (1 - eps) if s < 0 else eps
                like0 = eps if s < 0 else (1 - eps)
                logp += np.log(np.where(wi == 1, like1, like0))
        logp -= logp.max()
        post = np.exp(logp)
        return W, post / post.sum(), parents

    def _event_prob(self, traj, W, post):
        high = np.array([t.probability >= self.high for t in traj.turns])
        nh = int(high.sum())
        if nh == 0:
            return 0.0
        return float(post[W[:, high].sum(axis=1) >= label_threshold(nh, self.frac)].sum())

    def _em(self, trajs, eps):
        self.theta, self.theta_back = {}, {}
        for _ in range(self.em_iters):
            num, den = defaultdict(float), defaultdict(float)
            for traj in trajs:
                W, post, parents = self._posterior_configs(traj, eps)
                for i, turn in enumerate(traj.turns):
                    b = conf_bucket(turn.probability)
                    pw = W[:, parents[i]].max(axis=1) if parents[i] else np.zeros(len(W), dtype=np.int8)
                    for v in (0, 1):
                        m = pw == v
                        mass = post[m].sum()
                        if mass <= 0:
                            continue
                        err = post[m & (W[:, i] == 1)].sum()
                        for key in ((turn.attribute, b, v), (b, v)):
                            num[key] += err
                            den[key] += mass
            back = {k: float(np.clip(num[k] / den[k], 0.01, 0.99)) for k in den if len(k) == 2 and den[k] > 0}
            self.theta = {
                k: float(np.clip((num[k] + self.k * back.get((k[1], k[2]), 0.5)) / (den[k] + self.k), 0.01, 0.99))
                for k in den
                if len(k) == 3 and den[k] > 0
            }
            self.theta_back = back

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels, dtype=float)
        best = None
        for eps in EPS_GRID:
            self._em(trajectories, eps)
            pred = np.array([self._event_prob(t, *self._posterior_configs(t, eps)[:2]) for t in trajectories])
            loss = _bce(pred, y)
            if best is None or loss < best[0]:
                best = (loss, eps, dict(self.theta), dict(self.theta_back))
        _, self.eps, self.theta, self.theta_back = best
        return self

    def score(self, trajectory, dag=None) -> float:
        W, post, _ = self._posterior_configs(trajectory, self.eps)
        return self._event_prob(trajectory, W, post)


# ------------------------------------------------- supervised evidence model


def evidence_features(traj) -> dict:
    """Discrete evidence for the final-commitment event (dbbench)."""
    f = {}
    for turn in traj.turns:
        for name, v in turn.signals.items():
            if name == "resample_agree":
                continue  # deterministic function of the agreement count below
            f[f"{turn.attribute}:{name}"] = int(v)
    last = traj.turns[-1]
    c = agreement_count(last)
    f["final:agree"] = -1 if c is None else int(c)
    f["final:pbucket"] = int(min(last.probability, 0.999) * 10)  # deciles
    return f


class EvidenceNB(TrajectoryScorer):
    name = "pge_evidence_nb"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels, dtype=int)
        feats = [evidence_features(t) for t in trajectories]
        self.keys = sorted({k for f in feats for k in f})
        self.values = {k: sorted({f.get(k, "NA") for f in feats}, key=str) for k in self.keys}
        self.logprior = np.log(np.array([np.mean(y == 0), np.mean(y == 1)]) + 1e-9)
        self.table = {}
        for k in self.keys:
            vals = self.values[k]
            cnt = np.full((len(vals), 2), self.alpha)
            for f, yi in zip(feats, y):
                cnt[vals.index(f.get(k, "NA")), yi] += 1
            self.table[k] = (vals, np.log(cnt / cnt.sum(axis=0, keepdims=True)))
        return self

    def score(self, trajectory, dag=None) -> float:
        f = evidence_features(trajectory)
        lp = self.logprior.copy()
        for k in self.keys:
            vals, tab = self.table[k]
            v = f.get(k, "NA")
            if v in vals:
                lp += tab[vals.index(v)]
        lp -= lp.max()
        p = np.exp(lp)
        return float(p[1] / p.sum())
