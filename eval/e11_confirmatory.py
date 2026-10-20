"""e11: confirmatory analysis. Implements results/PREREG_confirmatory.md ($0).

    python3 -m eval.e11_confirmatory            # confirmatory corpora (seed 7)
    python3 -m eval.e11_confirmatory --dev      # same analysis on the development
                                                # corpora (seed 42), labelled as such

Per corpus and label (L_U primary, L secondary): every pre-registered detector
is fit on the train split and scored once on the test split of split seeds
42/123/456 (e6 protocol, eval.e7_common.run_protocol). Differences use the
paired bootstrap clustered by session (eval.e7_common.cluster_boot_diff).
Pooled-over-agents estimates use a stratified cluster bootstrap (sessions
resampled within each agent). The claim column applies the wording rules of
the pre-registration mechanically.

Writes results/e11_confirmatory/{summary.md, numbers.json} (or dev_*).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e7_common import SEEDS, cluster_boot_diff, run_protocol
from eval.score_trajectories import DAG_BUILDERS, build_dag, split_task_ids
from pge.baselines.base import TrajectoryScorer
from pge.baselines.confidence import HighConfContradicted, MeanConfidence, NegativeSignalFraction
from pge.baselines.context_lr import ContextLR, PriorOnly
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.baselines.gcn import GCNClassifier
from pge.baselines.posterior import (
    ChannelPosterior, IndependentPosterior, label_threshold, poisson_binomial_tail, turn_signal,
)
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import SemanticEntropy
from pge.benchmarks import healthcare_llmpatient
from pge.benchmarks.healthcare_v2 import (
    USER_DEPENDENT_SCHEMA, AgentGuessV2, derive_ground_truth_v2, eval_label_user, sample_session_v2,
    user_dependent_attrs,
)
from pge.graphon import GraphonEstimator
from pge.trajectory import load_jsonl

DAG_BUILDERS.setdefault(healthcare_llmpatient.DOMAIN, healthcare_llmpatient.dag_from_trajectory)

REPO = Path(__file__).resolve().parent.parent
T = REPO / "data" / "trajectories"
CONFIRM = [  # (tag, path, generation seed, n)
    ("70B", T / "healthcare_confirm" / "70B_7.jsonl", 7, 600),
    ("8B", T / "healthcare_confirm" / "8B_7.jsonl", 7, 600),
    ("qwen72b", T / "healthcare_confirm" / "qwen72b_7.jsonl", 7, 600),
    ("gpt-4o-mini", T / "healthcare_confirm" / "gpt-4o-mini_7.jsonl", 7, 600),
    ("mistral-small-24b", T / "healthcare_confirm" / "mistral-small-24b_7.jsonl", 7, 600),
]
PATIENT = [
    ("70B LLM-patient (profile)", T / "healthcare_llmpatient" / "70B_7.jsonl", 7, 600),
    ("70B LLM-patient (informed)", T / "healthcare_llmpatient" / "70B_7_informed.jsonl", 7, 600),
]
DEV = [
    ("70B (dev)", T / "healthcare_v2" / "70B_42.jsonl", 42, 600),
    ("8B (dev)", T / "healthcare_v2" / "8B_42.jsonl", 42, 200),
]
COVE = REPO / "results" / "cove_baseline" / "70B_on_70B_7" / "scores.json"
K10 = T / "healthcare_confirm_k10" / "70B_7_k10.jsonl"
CLAR = REPO / "results" / "e8_closed_loop" / "70B_7" / "summary.txt"
MIN_SESSIONS = 300
N_BOOT = 2000
GRAPHON_VARIANTS = [("position", "abs"), ("degree", "abs"), ("spectral", "abs"),
                    ("feature", "abs"), ("feature", "signed")]


# ------------------------------------------------------------ detector wrappers


class SampleRestricted(TrajectoryScorer):
    """Sampling detector computed only on turns whose attribute is in `attrs`."""

    def __init__(self, det, attrs):
        self.det, self.attrs = det, frozenset(attrs)
        self.name = det.name

    def score(self, trajectory, dag=None):
        t2 = copy.deepcopy(trajectory)
        for turn in t2.turns:
            if turn.attribute not in self.attrs:
                turn.samples = []
        return self.det.score(t2, dag)


class Oriented(TrajectoryScorer):
    """Zero-shot score whose sign is chosen on the train split (AUROC >= 0.5)."""

    def __init__(self, det, name):
        self.det, self.name, self.sign = det, name, 1.0

    def fit(self, trajectories, dags=None, labels=None):
        s = np.array([self.det.score(t) for t in trajectories])
        y = np.asarray(labels)
        if 0 < y.sum() < len(y):
            self.sign = 1.0 if auroc(s, y) >= 0.5 else -1.0
        return self

    def score(self, trajectory, dag=None):
        return self.sign * self.det.score(trajectory, dag)


class Graphon(TrajectoryScorer):
    """e6 graphon variant: fit on clean train DAGs (fallback all), score = -loglik."""

    def __init__(self, method, weighting, blocks=6):
        self.method, self.weighting, self.blocks = method, weighting, blocks
        self.name = f"graphon_{method}|{weighting}"
        self.est = None

    def fit(self, trajectories, dags=None, labels=None):
        y = np.asarray(labels)
        clean = [d for d, l in zip(dags, y) if l == 0]
        self.est = GraphonEstimator(B=self.blocks, block_method=self.method,
                                    weighting=self.weighting, normalize=True).fit(
            clean if len(clean) >= 5 else list(dags))
        return self

    def score(self, trajectory, dag=None):
        return -self.est.log_likelihood(dag if dag is not None else build_dag(trajectory))


def factories(label: str, full: bool = True):
    """Pre-registered detector set for label 'L_U' or 'L'."""
    ea = USER_DEPENDENT_SCHEMA if label == "L_U" else None
    sample_attrs = USER_DEPENDENT_SCHEMA if label == "L_U" else None

    def sampled(det):
        return SampleRestricted(det, sample_attrs) if sample_attrs is not None else det

    F = [
        ("V6", lambda: ChannelPosterior(event_attrs=ea)),
        ("V3", lambda: IndependentPosterior(by_value=True, event_attrs=ea)),
        ("V6-prior", lambda: PriorOnly(ChannelPosterior(event_attrs=ea))),
        ("V3-prior", lambda: PriorOnly(IndependentPosterior(by_value=True, event_attrs=ea))),
        ("ctx_lr", lambda: ContextLR()),
        ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
        ("highconf_contradicted", HighConfContradicted),
        ("neg_signal_fraction", NegativeSignalFraction),
        ("dag_logreg", DagFeatureClassifier),
        ("mean_confidence (oriented)", lambda: Oriented(MeanConfidence(), "mean_confidence (oriented)")),
    ]
    if full:
        F += [
            ("SelfCheckGPT", lambda: sampled(SelfCheckGPT())),
            ("semantic_entropy", lambda: sampled(SemanticEntropy())),
            ("gcn", GCNClassifier),
        ] + [(f"graphon_{m}|{w}", lambda m=m, w=w: Graphon(m, w)) for m, w in GRAPHON_VARIANTS]
    return F


# ------------------------------------------------------------------ data


def load(path: Path, seed: int, n: int):
    trajs = load_jsonl(path)
    rng = np.random.default_rng(seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(n)}
    L, LU = [], []
    for t in trajs:
        s = sessions[t.task_id]
        assert [x.attribute for x in t.turns] == s.schedule, f"{path.name} {t.task_id}: stream mismatch"
        g = {x.attribute: AgentGuessV2(x.attribute, x.guess_value, x.probability) for x in t.turns}
        lu = eval_label_user(g, s)
        if "eval_label_user" in t.meta:
            assert t.meta["eval_label_user"] == lu, f"{t.task_id}: stored L_U mismatch"
        L.append(t.eval_label)
        LU.append(lu)
    load.sessions = sessions
    return trajs, np.array(L, dtype=int), np.array(LU, dtype=int)


def add_external(R, name, score_by_tid):
    """Insert a precomputed zero-shot score (e.g. CoVe) aligned with R's test sets."""
    if not all(t in score_by_tid for tids in R["tid"] for t in tids):
        return False
    R["scores"][name] = [np.array([score_by_tid[t] for t in tids]) for tids in R["tid"]]
    R["auroc"][name] = [auroc(s, y) for s, y in zip(R["scores"][name], R["y"])]
    return True


# ------------------------------------------------------------- statistics


def mean_auc(R, d):
    return float(np.mean(R["auroc"][d]))


def pooled_rows(R, d):
    return np.concatenate(R["scores"][d])


def strat_boot(Rs: list, a: str, b: str | None, n=N_BOOT, seed=0):
    """Stratified cluster bootstrap over agents: mean over agents of
    AUROC(a) - AUROC(b) (or AUROC(a) if b is None) on pooled test rows,
    sessions resampled within each agent. Returns (point, lo, hi) where point
    is the mean over agents of the per-agent seed-averaged value."""
    point = float(np.mean([mean_auc(R, a) - (mean_auc(R, b) if b else 0.0) for R in Rs]))
    rng = np.random.default_rng(seed)
    prep = []
    for R in Rs:
        y = np.concatenate(R["y"])
        _, inv = np.unique(np.concatenate(R["tid"]), return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
        prep.append((pooled_rows(R, a), pooled_rows(R, b) if b else None, y, members))
    vals = []
    for _ in range(n):
        per = []
        for sa, sb, y, members in prep:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            yb = y[idx]
            if yb.sum() in (0, len(idx)):
                break
            per.append(auroc(sa[idx], yb) - (auroc(sb[idx], yb) if sb is not None else 0.0))
        else:
            vals.append(np.mean(per))
    vals = np.array(vals)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(ci, signed=True):
    m, lo, hi = ci
    s = "*" if signed and (lo > 0 or hi < 0) else ""
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]{s}"


def learning_curve(trajs, y, dags, sizes=(25, 50, 100, 200), draws=5):
    """Amendment 1 (descriptive): test AUROC vs number of labelled training
    sessions for V6, ctx_lr, ctx_lr+fb. Train subsets drawn from each split
    seed's train split (both classes required), test split unchanged."""
    ids = [t.task_id for t in trajs]
    make = {"V6": lambda lab: ChannelPosterior(event_attrs=lab), "ctx_lr": lambda lab: ContextLR(),
            "ctx_lr+fb": lambda lab: ContextLR(with_feedback=True)}
    out = {}
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids(ids, 0.7, seed)
        tr = np.array([i for i, t in enumerate(ids) if t in tr_ids])
        te = [i for i, t in enumerate(ids) if t in te_ids]
        for n in list(sizes) + [len(tr)]:
            for d in range(draws if n < len(tr) else 1):
                rng = np.random.default_rng((seed, n, d))
                for _ in range(50):
                    sub = tr if n >= len(tr) else rng.choice(tr, n, replace=False)
                    if 0 < y[sub].sum() < len(sub):
                        break
                else:
                    continue
                for name, mk in make.items():
                    det = mk(LC_EVENT[0]).fit([trajs[i] for i in sub], [dags[i] for i in sub], y[sub])
                    s = np.array([det.score(trajs[i], dags[i]) for i in te])
                    out.setdefault((name, "all" if n >= len(tr) else n), []).append(auroc(s, y[te]))
    return {k: float(np.mean(v)) for k, v in out.items()}


LC_EVENT = [None]  # event_attrs for the learning-curve V6 (set per label)


def claim(hid, ci):
    m, lo, hi = ci
    if hid == "H1":
        return ("sampling detectors are near chance on user-specific hallucination"
                if lo >= 0.40 and hi <= 0.60 else "report the AUROC; do not say 'near chance'")
    pos = lo > 0
    return {
        "H2": ("the user's implicit reactions add detectable signal beyond population base rates" if pos else
               "at eps = 0.25 with partial feedback, reactions add no detectable signal beyond base rates"),
        "H3": ("PGE beats the strongest context-only detector" if pos else
               "context alone matches PGE on this label"),
        "H4": ("PGE beats sampling-consistency detectors" if pos else "as measured (no superiority claim)"),
        "H2b": ("the user's reactions add detectable signal beyond the commitments themselves "
                "(supervised detector)" if pos else "no detectable feedback increment for the supervised detector"),
        "H4b": ("detectors on PGE observables beat sampling-consistency detectors" if pos else
                "as measured (no superiority claim)"),
    }[hid]


# --------------------------------------- Amendment 2: rules-aware analyses
# ORACLE INFORMATION (the simulator's clinic rules and its prior over hidden
# patient facts), never available to a deployed detector. Used to measure how
# much of L_U is predictable without the user (eta0, the context-only Bayes
# predictor, Theorem 1's bound made concrete) and how much the user's reactions
# add (etaF, Bayes with the true eps channel); and to compare detectors given
# the same per-session information (which commitments concern hidden facts).

import itertools as _it  # noqa: E402
import math as _math  # noqa: E402

_HID = [((d, s, k, a), (1 / 31) * (0.7 if s == "mild" else 0.3) * (0.8 if k == "normal" else 0.2)
         * (0.65 if a == "low" else 0.35))
        for d, s, k, a in _it.product(range(31), ("mild", "severe"), ("normal", "impaired"), ("low", "high"))]


def bayes_LU(traj, setup, eps=None):
    """P(L_U = 1 | intake, commitments[, feedback]) under the simulator's
    hidden-fact prior; with eps, the symmetric channel on reactions to
    user-dependent commitments (the others' likelihood does not depend on the
    hidden facts and cancels)."""
    U = user_dependent_attrs(setup)
    H = [t for t in traj.turns if t.probability >= 0.6 and t.attribute in U]
    if not H:
        return 0.0
    k = _math.ceil(0.5 * len(H))
    obs = [(t, turn_signal(t)) for t in traj.turns if t.attribute in U] if eps is not None else []
    num = den = 0.0
    for (d, sv, kd, a), w in _HID:
        tr = derive_ground_truth_v2(setup, {"days_since_treatment": d, "allergy_severity": sv,
                                            "kidney_function": kd, "anxiety_level": a})
        lab = sum(t.guess_value != tr[t.attribute] for t in H) >= k
        lik = 1.0
        for t, sig in obs:
            if sig != 0:
                lik *= (1 - eps) if sig == (1 if t.guess_value == tr[t.attribute] else -1) else eps
        num += w * lik * lab
        den += w * lik
    return num / den


class Fixed(TrajectoryScorer):
    def __init__(self, name, table):
        self.name, self.table = name, table

    def score(self, trajectory, dag=None):
        return self.table[trajectory.task_id]


class V6Session(ChannelPosterior):
    """V6 with the label event over each session's exact user-dependent set."""
    U: dict = {}

    def __init__(self):
        super().__init__()
        self.name = "V6-session"

    def _score(self, traj):
        Ut = self.U[traj.task_id]
        q = [self._q_turn(t) for t in traj.turns if t.probability >= self.high and t.attribute in Ut]
        return poisson_binomial_tail(np.array(q), label_threshold(len(q), self.frac)) if q else 0.0


class LRSession(ContextLR):
    """ctx_lr(+fb) given the same per-session information: cells only for
    confident commitments in the session's user-dependent set, plus |H_U|."""
    U: dict = {}

    def __init__(self, with_feedback=False):
        super().__init__(with_feedback=with_feedback)
        self.name = "LR-session" + ("+fb" if with_feedback else "")

    def _features(self, traj):
        Ut = self.U[traj.task_id]
        x = np.zeros(len(self.vocab))
        for turn in traj.turns:
            if turn.probability < self.high or turn.attribute not in Ut:
                continue
            for k in self._keys(turn):
                if k in self.vocab:
                    x[self.vocab[k]] += 1
        hu = sum(t.probability >= self.high and t.attribute in Ut for t in traj.turns)
        return np.concatenate([x, [hu, len(traj.turns), float(np.mean([t.probability for t in traj.turns]))]])


def rules_aware(trajs, sessions, y, dags, eps):
    """Amendment 2 detectors on L_U; eps=None (unknown channel) skips etaF."""
    U = {t.task_id: user_dependent_attrs(sessions[t.task_id].setup_answers) for t in trajs}
    V6Session.U = LRSession.U = U
    e0 = {t.task_id: bayes_LU(t, sessions[t.task_id].setup_answers) for t in trajs}
    F = [("eta0 (Bayes, no feedback)", lambda: Fixed("eta0", e0)),
         ("V6-session", V6Session), ("LR-session", LRSession), ("LR-session+fb", lambda: LRSession(True))]
    if eps is not None:
        eF = {t.task_id: bayes_LU(t, sessions[t.task_id].setup_answers, eps) for t in trajs}
        F.insert(1, ("etaF (Bayes, feedback)", lambda: Fixed("etaF", eF)))
    return run_protocol(trajs, y, F, dags=dags)


# ------------------------------------------------------------------ report


TABLE_DETS = ["V6", "V3", "V6-prior", "ctx_lr", "ctx_lr+fb", "SelfCheckGPT", "semantic_entropy",
              "highconf_contradicted", "neg_signal_fraction", "dag_logreg", "gcn",
              "mean_confidence (oriented)", "best graphon", "cove"]


def best_graphon(R):
    names = [d for d in R["auroc"] if d.startswith("graphon_")]
    if not names:
        return None
    b = max(names, key=lambda d: mean_auc(R, d))
    R["auroc"]["best graphon"], R["scores"]["best graphon"] = R["auroc"][b], R["scores"][b]
    return b


def corpus_block(tag, R, label, lines, numbers):
    dets = [d for d in TABLE_DETS if d in R["auroc"]]
    lines.append(f"| {tag} | " + " | ".join(f"{mean_auc(R, d):.3f}" for d in dets) + " |")
    numbers.setdefault(label, {}).setdefault(tag, {})["auroc"] = {d: mean_auc(R, d) for d in dets}
    numbers[label][tag]["auroc_per_seed"] = {d: [float(x) for x in R["auroc"][d]] for d in dets}
    numbers[label][tag]["pos_rate"] = float(np.concatenate(R["y"]).mean())
    return dets


def per_agent_diffs(tag, R, label, lines, numbers):
    y = np.concatenate(R["y"])
    g = np.concatenate(R["tid"])
    comps = [("V6", "V6-prior"), ("V6", "ctx_lr"), ("V6", "ctx_lr+fb"), ("V6", "SelfCheckGPT"),
             ("V6", "semantic_entropy"), ("V6", "highconf_contradicted"), ("V6", "V3"),
             ("ctx_lr+fb", "ctx_lr"), ("V6", "cove")]
    row = []
    for a, b in comps:
        if a in R["scores"] and b in R["scores"]:
            ci = cluster_boot_diff(pooled_rows(R, a), pooled_rows(R, b), y, g, n=N_BOOT)
            numbers[label][tag].setdefault("diffs", {})[f"{a} - {b}"] = ci
            row.append(f"{a} - {b}: {fmt(ci)}")
    lines.append(f"- **{tag}**: " + "; ".join(row))


def analyse(corpora, label_sets, out_name, full=True):
    t0 = time.time()
    lines, numbers = [], {}
    runs = {lab: {} for lab in ("L_U", "L")}
    rules = {}
    for tag, path, seed, n in corpora:
        if not path.exists():
            lines.append(f"> {tag}: `{path.relative_to(REPO)}` not found; skipped.")
            continue
        trajs, L, LU = load(path, seed, n)
        dags = [build_dag(t) for t in trajs]
        sessions = load.sessions
        if "L_U" in label_sets:
            eps = None if any(t.domain == healthcare_llmpatient.DOMAIN for t in trajs) else \
                float(trajs[0].meta.get("epsilon", 0.25))
            rules[tag] = (rules_aware(trajs, sessions, LU, dags, eps), len(trajs))
        nfail = sum(len(t.meta.get("parse_failed_turns", [])) for t in trajs)
        print(f"[{tag}] n={len(trajs)} L={L.mean():.3f} L_U={LU.mean():.3f} parse-failed turns={nfail}",
              flush=True)
        numbers.setdefault("corpora", {})[tag] = {"n": len(trajs), "L": float(L.mean()),
                                                  "L_U": float(LU.mean()), "parse_failed_turns": nfail}
        for lab, y in (("L_U", LU), ("L", L)):
            if lab not in label_sets:
                continue
            R = run_protocol(trajs, y, factories(lab, full=full), dags=dags)
            best_graphon(R)
            if tag.startswith("70B") and COVE.exists() and "dev" not in tag and "patient" not in tag:
                cove = json.loads(COVE.read_text())["score_by_task"]
                add_external(R, "cove", cove)
            runs[lab][tag] = (R, len(trajs))
            if full and len(trajs) >= MIN_SESSIONS:
                LC_EVENT[0] = USER_DEPENDENT_SCHEMA if lab == "L_U" else None
                numbers.setdefault("learning_curve", {}).setdefault(lab, {})[tag] = {
                    f"{k[0]}@{k[1]}": v for k, v in learning_curve(trajs, y, dags).items()}
            print(f"  [{lab}] V6 {mean_auc(R, 'V6'):.3f} V6-prior {mean_auc(R, 'V6-prior'):.3f} "
                  f"ctx_lr {mean_auc(R, 'ctx_lr'):.3f} ({time.time() - t0:.0f}s)", flush=True)

    for lab, title in (("L_U", "user-specific label L_U (primary)"), ("L", "composite label L (secondary)")):
        if not runs[lab]:
            continue
        lines += ["", f"## {title}", ""]
        first = next(iter(runs[lab].values()))[0]
        dets = [d for d in TABLE_DETS if d in first["auroc"]]
        hdr = ["corpus"] + dets
        lines += ["Session AUROC, mean over split seeds 42/123/456:", "",
                  "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for tag, (R, _) in runs[lab].items():
            corpus_block(tag, R, lab, lines, numbers)
        lines += ["", "Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:", ""]
        for tag, (R, _) in runs[lab].items():
            per_agent_diffs(tag, R, lab, lines, numbers)

        lc = numbers.get("learning_curve", {}).get(lab, {})
        if lc:
            lines += ["", "Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions", "",
                      "| corpus | detector | 25 | 50 | 100 | 200 | all train |", "|---|---|---|---|---|---|---|"]
            for tag, d in lc.items():
                for name in ("V6", "ctx_lr", "ctx_lr+fb"):
                    lines.append(f"| {tag} | {name} | " + " | ".join(
                        f"{d.get(f'{name}@{n}', float('nan')):.3f}" for n in (25, 50, 100, 200, "all")) + " |")
        pool = [R for tag, (R, n) in runs[lab].items() if n >= MIN_SESSIONS]
        if len(pool) >= 2:
            lines += ["", f"Pooled over {len(pool)} corpora (stratified cluster bootstrap):", "",
                      "| id | quantity | estimate [95% CI] | licensed claim |", "|---|---|---|---|"]
            H = [("H1", "SelfCheckGPT", None), ("H1", "semantic_entropy", None),
                 ("H2", "V6", "V6-prior"), ("H3", "V6", "ctx_lr"),
                 ("H4", "V6", "SelfCheckGPT"), ("H4", "V6", "semantic_entropy"),
                 ("H2b", "ctx_lr+fb", "ctx_lr"), ("H4b", "ctx_lr+fb", "SelfCheckGPT"),
                 ("H4b", "ctx_lr+fb", "semantic_entropy"),
                 ("sec.", "V6", "ctx_lr+fb"), ("sec.", "V6", "V3")]
            for hid, a, b in H:
                if not all(a in R["scores"] and (b is None or b in R["scores"]) for R in pool):
                    continue
                ci = strat_boot(pool, a, b)
                q = f"AUROC({a})" if b is None else f"{a} - {b}"
                c = claim(hid, ci) if hid != "sec." else "reported as is"
                lines.append(f"| {hid} | {q} | {fmt(ci, signed=b is not None)} | {c} |")
                numbers[lab].setdefault("pooled", {})[f"{hid}: {q}"] = ci
    if rules:
        lines += ["", "## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and "
                  "hidden-fact prior; not deployable)", "",
                  "| corpus | eta0 (Bayes, no feedback) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |",
                  "|---|---|---|---|---|---|"]
        for tag, (R, _) in rules.items():
            cells = [f"{mean_auc(R, d):.3f}" if d in R["auroc"] else "n/a" for d in
                     ("eta0 (Bayes, no feedback)", "etaF (Bayes, feedback)", "V6-session", "LR-session", "LR-session+fb")]
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
            numbers.setdefault("rules_aware", {})[tag] = {d: mean_auc(R, d) for d in R["auroc"]}
        lines += ["", "Differences per corpus (session-clustered):", ""]
        comps = [("etaF (Bayes, feedback)", "eta0 (Bayes, no feedback)"), ("V6-session", "LR-session+fb"),
                 ("V6-session", "eta0 (Bayes, no feedback)"), ("LR-session+fb", "LR-session")]
        for tag, (R, _) in rules.items():
            y = np.concatenate(R["y"]); g = np.concatenate(R["tid"]); row = []
            for a, b in comps:
                if a in R["scores"] and b in R["scores"]:
                    ci = cluster_boot_diff(pooled_rows(R, a), pooled_rows(R, b), y, g, n=N_BOOT)
                    numbers["rules_aware"][tag][f"{a} - {b}"] = ci
                    row.append(f"{a} - {b}: {fmt(ci)}")
            lines.append(f"- **{tag}**: " + "; ".join(row))
        pool = [R for tag, (R, n) in rules.items() if n >= MIN_SESSIONS and "etaF (Bayes, feedback)" in R["auroc"]]
        if len(pool) >= 2:
            lines += ["", f"Pooled over {len(pool)} corpora:", "", "| id | quantity | estimate [95% CI] | licensed claim |",
                      "|---|---|---|---|"]
            for hid, a, b in [("H5", "etaF (Bayes, feedback)", "eta0 (Bayes, no feedback)"),
                              ("H6", "V6-session", "LR-session+fb"), ("sec.", "V6-session", "eta0 (Bayes, no feedback)")]:
                ci = strat_boot(pool, a, b)
                c = {"H5": ("the user's reactions carry information beyond the best possible context-only predictor"
                            if ci[1] > 0 else "no information beyond the context-only Bayes predictor detected"),
                     "H6": ("given the same information, the PGE posterior beats the supervised detector"
                            if ci[1] > 0 else "given the same information, the PGE posterior does not beat the supervised detector"),
                     "sec.": "reported as is"}[hid]
                lines.append(f"| {hid} | {a} - {b} | {fmt(ci)} | {c} |")
                numbers["rules_aware"].setdefault("pooled", {})[f"{hid}: {a} - {b}"] = ci
    return lines, numbers, time.time() - t0


def k10_block(lines, numbers):
    if not K10.exists():
        return
    trajs, L, LU = load(K10, 7, 600)
    by = {t.task_id: (t, l, lu) for t, l, lu in zip(trajs, L, LU)}
    te = sorted(split_task_ids(list(by), 0.7, 42)[1])
    lines += ["", "## k = 10 resamples (70B, split-seed-42 test sessions, user-dependent turns)", ""]
    for lab, idx in (("L_U", 2), ("L", 1)):
        y = np.array([by[t][idx] for t in te])
        res = {}
        for k in (3, 10):
            det = SampleRestricted(SelfCheckGPT(), USER_DEPENDENT_SCHEMA)
            s = []
            for t in te:
                tt = copy.deepcopy(by[t][0])
                for turn in tt.turns:
                    turn.samples = turn.samples[:k]
                s.append(det.score(tt))
            res[k] = auroc(np.array(s), y)
        lines.append(f"- {lab}: SelfCheckGPT k=3 {res[3]:.3f}, k=10 {res[10]:.3f} (n={len(te)})")
        numbers.setdefault("k10", {})[lab] = res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev", action="store_true", help="development corpora (seed 42), labelled as such")
    ap.add_argument("--quick", action="store_true", help="skip slow baselines (sampling, GCN, graphons)")
    args = ap.parse_args(argv)
    out_dir = REPO / "results" / "e11_confirmatory"
    prefix = "dev_" if args.dev else ""
    head = [f"# e11 {'DEVELOPMENT corpora (seed 42, protocol v2)' if args.dev else 'confirmatory corpora (seed 7, protocol v2fb)'}",
            "", "Implements results/PREREG_confirmatory.md. Fit on train, test once; split seeds 42/123/456.",
            "L_U detectors: V6/V3 label event restricted to USER_DEPENDENT_SCHEMA and trained on L_U;",
            "sampling baselines on user-dependent turns only. L detectors: as in e6.", ""]
    corpora = DEV if args.dev else CONFIRM
    lines, numbers, secs = analyse(corpora, ("L_U", "L"), prefix, full=not args.quick)
    if not args.dev:
        pl, pn, _ = analyse(PATIENT, ("L_U", "L"), prefix, full=False)
        if any("|" in x for x in pl):
            lines += ["", "# LLM-simulated patient (behavioral feedback)", ""] + pl
            numbers["llm_patient"] = pn
        k10_block(lines, numbers)
        if CLAR.exists():
            lines += ["", "# Measured closed-loop clarification (results/e8_closed_loop/70B_7/summary.txt)",
                      "", "```", CLAR.read_text().rstrip(), "```"]
    lines.append(f"\nruntime {secs:.0f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}summary.md").write_text("\n".join(head + lines) + "\n")
    (out_dir / f"{prefix}numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(head + lines))


if __name__ == "__main__":
    main()
