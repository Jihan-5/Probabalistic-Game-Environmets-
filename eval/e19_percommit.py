"""e19: per-commitment risk, computed per split seed with session-clustered CIs (exploratory).

    python3 -m eval.e19_percommit

For each confirmatory healthcare corpus and split seed (42/123/456): V6 and
the frozen V7 are fit on the training split; on the test split every
confident commitment is scored by V7's q_t, V6's q_t, the commitment's own
reaction, resample disagreement and stated confidence (sign chosen on training
commitments). Truth: the commitment is wrong for this user. Two evaluation
sets: (a) commitments whose truth is user-dependent in that session (needs the
simulator's rules; the set used in the paper's Table 1), (b) commitments on
the schema-level user-dependent attributes (knowable without the rules).
AUROC and 10-bin ECE are computed per seed and averaged over seeds; 95% CIs
resample test sessions (one draw of session ids applied to every seed and,
for the pooled row, to every agent, which share sessions). Writes
results/e19_percommit/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

import numpy as np

from eval.bootstrap_ci import auroc
from eval.e11_confirmatory import CONFIRM, REPO, build_dag, load
from eval.e12_exploratory import ece
from eval.e7_common import SEEDS, TRAIN_FRAC
from eval.score_trajectories import split_task_ids
from pge.baselines.intake_posterior import IntakePosterior, v7
from pge.baselines.posterior import ChannelPosterior, agreement_count, turn_signal

OUT = REPO / "results" / "e19_percommit"
SCORES = ("V7", "V6", "reaction", "disagreement", "stated_conf")
N_BOOT = 1000


def rows_for(trajs, LU, sessions, dom):
    """{seed: {set: list of (session id, wrong, {score: value})}}"""
    ids = [t.task_id for t in trajs]
    S = dom.USER_DEPENDENT_SCHEMA
    out = {}
    for seed in SEEDS:
        tr_ids, te_ids = split_task_ids(ids, TRAIN_FRAC, seed)
        tr = [t for t in trajs if t.task_id in tr_ids]
        ytr = [LU[i] for i, t in enumerate(trajs) if t.task_id in tr_ids]
        d7 = v7(dom, "L_U").fit(tr, None, ytr)
        d6 = ChannelPosterior(event_attrs=S).fit(tr, [build_dag(t) for t in tr], ytr)
        cp, cw = [], []
        for t in tr:
            st = sessions[t.task_id].user_state
            for turn in t.turns:
                if turn.probability >= 0.6 and turn.attribute in S:
                    cp.append(turn.probability)
                    cw.append(int(turn.guess_value != st[turn.attribute]))
        sign = -1.0 if auroc(-np.array(cp), np.array(cw)) >= 0.5 else 1.0
        sets = {"user_dependent": [], "schema": []}
        for t in trajs:
            if t.task_id not in te_ids:
                continue
            s = sessions[t.task_id]
            U = dom.user_dependent_attrs(s.setup_answers)
            q7, q6 = d7.turn_q(t), d6.turn_q(t)
            for k, turn in enumerate(t.turns):
                if turn.probability < 0.6 or turn.attribute not in S:
                    continue
                c = agreement_count(turn)
                sc = {"V7": float(q7[k]), "V6": float(q6[k]), "reaction": -float(turn_signal(turn)),
                      "disagreement": 1.0 - c / len(turn.samples) if c is not None else 0.0,
                      "stated_conf": sign * turn.probability}
                r = (t.task_id, int(turn.guess_value != s.user_state[turn.attribute]), sc)
                sets["schema"].append(r)
                if turn.attribute in U:
                    sets["user_dependent"].append(r)
        out[seed] = sets
    return out


def to_arrays(rows, sid_index):
    return (np.array([sid_index[r[0]] for r in rows], dtype=int), np.array([r[1] for r in rows]),
            np.array([[r[2][k] for k in SCORES] for r in rows]))


def metrics(arrs, counts=None):
    """Mean over seeds of per-seed AUROC (ECE for the posteriors). counts: bootstrap multiplicity per session."""
    acc = {k: [] for k in SCORES}
    eac = {"V7": [], "V6": []}
    for sidx, y, X in arrs:
        if counts is not None:
            idx = np.repeat(np.arange(len(y)), counts[sidx])
            y_, X_ = y[idx], X[idx]
        else:
            y_, X_ = y, X
        if not 0 < y_.sum() < len(y_):
            continue
        for j, k in enumerate(SCORES):
            acc[k].append(auroc(X_[:, j], y_))
            if k in eac:
                eac[k].append(ece(X_[:, j], y_))
    res = {f"auroc_{k}": float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    res.update({f"ece_{k}": float(np.mean(v)) for k, v in eac.items() if v})
    return res


def main():
    t0 = time.time()
    per_agent, numbers = {}, {}
    for tag, path, seed, n in CONFIRM:
        if not path.exists():
            continue
        trajs, L, LU = load(path, seed, n)
        sessions, dom = load.sessions, load.domain
        IntakePosterior.INTAKE = {t.task_id: sessions[t.task_id].setup_answers for t in trajs}
        per_agent[tag] = rows_for(trajs, LU, sessions, dom)
        print(tag, f"{time.time() - t0:.0f}s", flush=True)
    lines = ["# e19: per-commitment risk, per split seed, session-clustered CIs (exploratory)", ""]
    sids = sorted({r[0] for d in per_agent.values() for s in d.values() for r in s["schema"]})
    sid_index = {sid: i for i, sid in enumerate(sids)}
    rng = np.random.default_rng(0)
    for which in ("user_dependent", "schema"):
        arrs = {tag: [to_arrays(d[s][which], sid_index) for s in SEEDS] for tag, d in per_agent.items()}
        point = {tag: metrics(a) for tag, a in arrs.items()}
        boots = {tag: [] for tag in per_agent}
        pooled_boot = []
        for _ in range(N_BOOT):
            counts = np.bincount(rng.integers(0, len(sids), len(sids)), minlength=len(sids))
            ms = {tag: metrics(a, counts) for tag, a in arrs.items()}
            for tag in per_agent:
                boots[tag].append(ms[tag])
            pooled_boot.append({k: float(np.mean([ms[t][k] for t in per_agent])) for k in ms[next(iter(ms))]})
        numbers[which] = {}
        keys = list(next(iter(point.values())).keys())
        lines += [f"## evaluation set: {which}", "", "| corpus | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for tag in per_agent:
            ci = {k: (point[tag][k], float(np.nanpercentile([b[k] for b in boots[tag]], 2.5)),
                      float(np.nanpercentile([b[k] for b in boots[tag]], 97.5))) for k in keys}
            numbers[which][tag] = ci
            lines.append(f"| {tag} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for v in ci.values()) + " |")
        pp = {k: float(np.mean([point[t][k] for t in per_agent])) for k in keys}
        pci = {k: (pp[k], float(np.nanpercentile([b[k] for b in pooled_boot], 2.5)),
                   float(np.nanpercentile([b[k] for b in pooled_boot], 97.5))) for k in keys}
        dif = [b["auroc_V7"] - b["auroc_reaction"] for b in pooled_boot]
        pci["V7 - reaction"] = (pp["auroc_V7"] - pp["auroc_reaction"], float(np.percentile(dif, 2.5)),
                                float(np.percentile(dif, 97.5)))
        dif = [b["auroc_V7"] - b["auroc_V6"] for b in pooled_boot]
        pci["V7 - V6"] = (pp["auroc_V7"] - pp["auroc_V6"], float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5)))
        numbers[which]["pooled"] = pci
        lines.append(f"| pooled | " + " | ".join(f"{pci[k][0]:.3f} [{pci[k][1]:.3f}, {pci[k][2]:.3f}]" for k in keys) + " |")
        lines += ["", f"- pooled V7 - reaction: {pci['V7 - reaction'][0]:+.3f} [{pci['V7 - reaction'][1]:+.3f}, {pci['V7 - reaction'][2]:+.3f}]",
                  f"- pooled V7 - V6: {pci['V7 - V6'][0]:+.3f} [{pci['V7 - V6'][1]:+.3f}, {pci['V7 - V6'][2]:+.3f}]", ""]
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
