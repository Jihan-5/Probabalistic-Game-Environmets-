"""Closed-loop clarification, MEASURED with the real agent (PAID).

Paid counterpart of eval/e8_clarification.py (offline simulation). Healthcare
70B corpus, test split of split seed 42 (split_task_ids(ids, 0.7, 42)),
identical sessions regenerated from the sample_session_v2 stream; every rerun
of session hcv2_<i> draws its eps-flip feedback from
default_rng((seed, 7919, i)) (the e5_intervention per-session noise stream),
fresh per condition, so all conditions face the same patient.

Conditions per test session:
  cached              the original cached session (no rerun, $0)
  detailed_questions  (A) the agent's context contains, from turn 0, the
                      patient's true answers for every UNDERDETERMINED
                      attribute in the session (patient-stated sentences);
                      whole session rerun. burden = number of answers.
  posterior@b         (B, b in --budgets) pick the b confident (p >= 0.6)
                      cached commitments with the highest posterior error q_t
                      (IndependentPosterior(by_value=True) fit on the seed-42
                      train split; q_t from eval.e8_clarification.
                      turn_posteriors). Turns before the earliest pick are
                      kept as cached; the agent re-runs every turn from there
                      on. From the turn of each pick onward the context holds
                      the patient's true answer for that attribute.
  no_question@b       (C) rerun from the same position as posterior@b with no
                      answer: the regeneration-noise control. Sessions with
                      no confident commitment get no question and no rerun in
                      B and C (cached commitments are kept).
  rerun_full          whole session rerun with no answers: the
                      regeneration-noise control for detailed_questions.
  random@b, lowconf@b (b in --selection-budgets) same as posterior@b but the b
                      confident commitments are chosen at random (per-session
                      stream default_rng((split_seed, 31, i))) or by lowest
                      stated confidence, as in e8: does the posterior's choice
                      of WHICH commitment to confirm matter?
Outcome per session and condition: eval_label_v2 on the resulting
commitments (cached prefix + regenerated suffix), wrong confident
commitments vs noise-free truth, number of questions, rerun turns, cost.

Usage:
    python3 -m eval.paid.run_clarification_closed_loop --max-cost 5
Writes results/e8_closed_loop/<model>_<seed>/{<condition>.jsonl, summary.txt, sessions.json}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from eval.e8_clarification import confident, turn_posteriors
from eval.generate_healthcare_v2 import SYSTEM_PROMPT_HC
from eval.paid.common import (
    ALLOWED_VALUES, REPO, add_dag_node, add_model_args, add_run_args, check_mock_out,
    is_failed, ledger_line, make_client, preflight, print_estimate, rel, run_parallel,
    spec_from_args, write_run_json,
)
from eval.score_trajectories import split_task_ids
from pge.agents.paid_client import CostMeter, RoutedAgent
from pge.baselines.posterior import IndependentPosterior
from pge.benchmarks.healthcare_llmpatient import STATEMENTS
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2, NO_FEEDBACK_ATTRS, UNDERDETERMINED, AgentGuessV2, eval_label_v2,
    sample_session_v2,
)
from pge.dag import ChainOfThoughtDAG
from pge.trajectory import Trajectory, Turn, load_jsonl, save_jsonl

HIGH = 0.6
NOISE_SALT = 7919  # eval/e5_intervention.py per-session noise stream
# e5_intervention anchor: $0.34 per 54 full-session reruns of ~9.85 turns = $0.00064/turn
# = ~726 tokens/turn at $0.88/M (incl. format retries).
EST_PER_TURN = (700, 26)

_FP = [(r"\bfor you\b", "for me"), (r"\bhelp you\b", "help me"), (r"\bdrive you\b", "drive me"),
       (r"\byou will\b", "I will"), (r"\byou can\b", "I can"), (r"\byou should\b", "I should"),
       (r"\byou stay\b", "I stay"), (r"\byou\b", "I"), (r"\byour\b", "my")]


def answer_sentence(attribute: str, value: str) -> str:
    """The patient's first-person statement of the TRUE value."""
    s = STATEMENTS[attribute][value]
    for a, b in _FP:
        s = re.sub(a, b, s)
    return s[0].upper() + s[1:] + "."


def answers_block(session, attrs) -> str:
    """Prompt text with the patient's answers (truth) for `attrs`. Lives only
    in the agent's prompt; never stored in a trajectory field."""
    lines = [f'  - about {a}: "{answer_sentence(a, session.user_state[a])}"' for a in attrs]
    return ("The patient answered these questions directly (the patient's own statements):\n"
            + "\n".join(lines))


@dataclasses.dataclass
class Plan:
    condition: str
    start: int            # first regenerated turn (len(turns) = no rerun)
    ask: dict             # turn index -> attribute answered from that turn on (None key = upfront)
    upfront: list         # attributes answered before turn 0


def plans_for(traj, q, budgets, selection_budgets=(1,), split_seed=42) -> list[Plan]:
    n = len(traj.turns)
    plans = [Plan("cached", n, {}, [])]
    ud = [t.attribute for t in traj.turns if t.attribute in UNDERDETERMINED]
    plans.append(Plan("detailed_questions", 0, {}, ud))
    plans.append(Plan("rerun_full", 0, {}, []))
    conf = confident(traj)

    def asked(name, picks):
        picks = sorted(picks)
        return Plan(name, picks[0] if picks else n, {j: traj.turns[j].attribute for j in picks}, [])

    for b in budgets:
        post = asked(f"posterior@{b}", sorted(conf, key=lambda j: -q[j])[:b])
        plans += [post, Plan(f"no_question@{b}", post.start, {}, [])]
    pid = int(traj.task_id.rsplit("_", 1)[1])
    perm = [conf[k] for k in np.random.default_rng((split_seed, 31, pid)).permutation(len(conf))]
    for b in selection_budgets:
        plans.append(asked(f"random@{b}", perm[:b]))
        plans.append(asked(f"lowconf@{b}", sorted(conf, key=lambda j: traj.turns[j].probability)[:b]))
    return plans


def run_plan(plan: Plan, traj: Trajectory, session, client, seed: int, epsilon: float):
    """Returns (trajectory, agent or None). Truth is used only to (a) phrase
    the patient's answers in the prompt and (b) the eps channel + label,
    exactly as in the generator."""
    n = len(traj.turns)
    G = ChainOfThoughtDAG()
    node_idx: dict = {}
    turns: list[Turn] = []
    guesses: dict = {}
    for t in traj.turns[: plan.start]:  # cached prefix, unchanged
        turns.append(dataclasses.replace(t))
        guesses[t.attribute] = AgentGuessV2(t.attribute, t.guess_value, t.probability)
        add_dag_node(G, node_idx, t.attribute, t.guess_value, t.probability,
                     t.signals.get("noisy_feedback", 0), t.turn)
    agent, statuses = None, []
    if plan.start < n:
        agent = RoutedAgent(
            attribute_schedule=session.schedule, client=client, allowed_values=ALLOWED_VALUES,
            attribute_descriptions=ATTR_DESCRIPTIONS_V2, temperature=0.3, max_tokens=200,
            system_prompt=SYSTEM_PROMPT_HC, request_timeout=45.0,
        )
        rng = np.random.default_rng((seed, NOISE_SALT, session.profile_id))
        for t in range(plan.start, n):
            attribute = session.schedule[t]
            answered = list(plan.upfront) + [a for j, a in sorted(plan.ask.items()) if j <= t]
            setup = dict(session.setup_answers)
            if answered:
                setup["patient_answers"] = answers_block(session, answered)
            guess, raw = agent.guess_full(t, setup, G)
            statuses.append(agent.last_parse_status)
            value = str(guess.value).strip().lower()
            guesses[attribute] = AgentGuessV2(attribute, value, guess.probability)
            correct = value == session.user_state[attribute]
            signal = 0 if attribute in NO_FEEDBACK_ATTRS else (1 if correct else -1)
            if signal != 0 and rng.random() < epsilon:
                signal = -signal
            turns.append(Turn(attribute=attribute, guess_value=value, probability=guess.probability,
                              turn=t, raw_response=raw, signals={"noisy_feedback": signal}))
            add_dag_node(G, node_idx, attribute, value, guess.probability, signal, t)
    n_q = len(plan.upfront) + len(plan.ask)
    wrong_conf = sum(1 for g in guesses.values()
                     if g.probability >= HIGH and g.value != session.user_state[g.attribute])
    out = Trajectory(
        task_id=traj.task_id, domain="healthcare_v2", model=traj.model, seed=traj.seed, turns=turns,
        eval_label=eval_label_v2(guesses, session), label_source="programmatic",
        meta={
            "condition": plan.condition,
            "rerun_from": plan.start,
            "n_questions": n_q,
            # attributes and positions only; answer values (truth) are not stored
            "questions": ([{"turn": 0, "attribute": a, "upfront": True} for a in plan.upfront]
                          + [{"turn": j, "attribute": a} for j, a in sorted(plan.ask.items())]),
            "wrong_confident": wrong_conf,
            "label_before": traj.eval_label,
            "parse_status": statuses,
            "parse_failed_turns": [plan.start + k for k, s in enumerate(statuses) if is_failed(s)],
            "cost": round(agent.cost, 6) if agent else 0.0,
            "tokens_in": agent.input_tokens_used if agent else 0,
            "tokens_out": agent.output_tokens_used if agent else 0,
        },
    )
    return out


def paired_boot(a, b, n=2000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(results: dict, conditions: list, header: str) -> str:
    tids = sorted(tid for tid, r in results.items() if all(c in r for c in conditions))
    L = [header, f"paired sessions with every condition complete: {len(tids)}", "",
         f"{'condition':20s} {'questions/sess':>14s} {'hall. rate':>10s} {'wrong conf/sess':>15s} "
         f"{'rerun turns':>11s} {'cost':>8s}"]
    col = {}
    for c in conditions:
        rows = [results[t][c] for t in tids]
        lab = np.array([r.eval_label for r in rows])
        wc = np.array([r.meta["wrong_confident"] for r in rows])
        col[c] = (lab, wc)
        L.append(f"{c:20s} {np.mean([r.meta['n_questions'] for r in rows]):14.2f} {lab.mean():10.3f} "
                 f"{wc.mean():15.3f} {sum(len(r.turns) - r.meta['rerun_from'] for r in rows):11d} "
                 f"{sum(r.meta['cost'] for r in rows):8.3f}")
    comps = []
    for c in conditions:
        if c.startswith("posterior@"):
            b = c.split("@")[1]
            comps += [(c, f"no_question@{b}"), (c, "detailed_questions"), (c, "cached")]
    comps += [("detailed_questions", "cached")] + [
        (c, "cached") for c in conditions if c.startswith("no_question@")]
    comps += [("detailed_questions", "rerun_full"), ("rerun_full", "cached")]
    for c in conditions:
        if c.startswith(("random@", "lowconf@")):
            comps += [(f"posterior@{c.split('@')[1]}", c), (c, "cached")]
    L += ["", "## paired bootstrap over sessions (2000): A - B, negative = A better",
          f"{'A':20s} {'B':20s} {'d hall. rate':>28s} {'d wrong conf/sess':>28s}"]
    for a, b in comps:
        if a in col and b in col and tids:
            m, lo, hi = paired_boot(col[a][0], col[b][0])
            m2, lo2, hi2 = paired_boot(col[a][1], col[b][1])
            L.append(f"{a:20s} {b:20s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]".ljust(70)
                     + f"{m2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="70B")
    ap.add_argument("--seed", type=int, default=42, help="generation seed of the cache")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=600, help="must match the generation run")
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--selection-budgets", type=int, nargs="*", default=[1],
                    help="budgets for the random@b / lowconf@b selection controls")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test sessions (pilot)")
    ap.add_argument("--cache", default=None)
    add_run_args(ap, max_cost=5.0)
    args = ap.parse_args(argv)
    check_mock_out(args)

    spec = spec_from_args(args)
    cache = Path(args.cache) if args.cache else \
        REPO / "data" / "trajectories" / "healthcare_v2" / f"{spec.short_name}_{args.seed}.jsonl"
    out_dir = Path(args.out) if args.out else \
        REPO / "results" / "e8_closed_loop" / f"{spec.short_name}_{args.seed}"
    trajs = load_jsonl(cache)
    by_id = {t.task_id: t for t in trajs}
    if any(t.model != spec.model_id for t in trajs):
        raise SystemExit(f"cache model {trajs[0].model} != --model {spec.model_id}")

    rng = np.random.default_rng(args.seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(args.n)}
    for t in trajs:
        assert [x.attribute for x in t.turns] == sessions[t.task_id].schedule, f"{t.task_id}: stream mismatch"

    tr_ids, te_ids = split_task_ids(list(by_id), 0.7, args.split_seed)
    tr = sorted(tr_ids)
    det = IndependentPosterior(by_value=True).fit([by_id[t] for t in tr], None,
                                                  np.array([by_id[t].eval_label for t in tr]))
    test = sorted(te_ids, key=lambda s: int(s.split("_")[1]))
    if args.limit:
        test = test[: args.limit]
    plans = {tid: plans_for(by_id[tid], turn_posteriors(det, by_id[tid]), args.budgets,
                            args.selection_budgets, args.split_seed) for tid in test}
    conditions = [p.condition for p in plans[test[0]]]

    units = [(tid, p) for tid in test for p in plans[tid]]
    paid = [(tid, p) for tid, p in units if p.start < len(by_id[tid].turns)]
    rows = []
    for c in conditions:
        turns_c = sum(len(by_id[tid].turns) - p.start for tid, p in paid if p.condition == c)
        if turns_c:
            rows.append((f"{c} (turns)", spec, turns_c, *EST_PER_TURN))
    est = print_estimate(
        f"closed-loop clarification on {len(test)} test sessions (split seed {args.split_seed}, "
        f"posterior eps={det.eps}) -> {rel(out_dir)}  [anchor: e5_intervention $0.34/54 reruns]",
        rows, args.max_cost)
    if args.estimate_only:
        return

    partial = out_dir / "all.partial.jsonl"
    results: dict = {}
    if args.resume and partial.exists():
        for t in load_jsonl(partial):
            results.setdefault(t.task_id, {})[t.meta["condition"]] = t
        print(f"[resume] {sum(len(v) for v in results.values())} units from {rel(partial)}", flush=True)
    for tid in test:  # $0 conditions
        for p in plans[tid]:
            if p.start >= len(by_id[tid].turns):
                results.setdefault(tid, {})[p.condition] = run_plan(
                    p, by_id[tid], sessions[tid], None, args.seed, args.epsilon)
    todo = [(tid, p) for tid, p in paid if p.condition not in results.get(tid, {})]

    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "agent", args)
    preflight([client])

    def one(unit):
        tid, p = unit
        return run_plan(p, by_id[tid], sessions[tid], client, args.seed, args.epsilon)

    def on_result(unit, traj):
        results.setdefault(unit[0], {})[unit[1].condition] = traj

    def flat():
        return [results[t][c] for t in sorted(results) for c in results[t]]

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = run_parallel(todo, one, args.concurrency, meter, on_result, label="rerun",
                         checkpoint=lambda: save_jsonl(flat(), partial))
    complete = all(c in results.get(tid, {}) for tid in test for c in conditions)
    save_jsonl(flat(), partial)
    if complete:
        for c in conditions:
            save_jsonl([results[tid][c] for tid in test], out_dir / f"{c.replace('@', '_at')}.jsonl")
        partial.unlink()
    summ = meter.summary()
    header = (f"# closed-loop clarification (paid, measured) — {spec.model_id}, cache {rel(cache)}, "
              f"test split seed {args.split_seed} ({len(test)} sessions), posterior eps={det.eps}, "
              f"noise stream (seed, {NOISE_SALT}, i), cost ${summ['cost']:.3f}")
    text = summarize(results, conditions, header)
    (out_dir / "summary.txt").write_text(text + "\n")
    (out_dir / "sessions.json").write_text(json.dumps({
        tid: {c: {"label": r.eval_label, "wrong_confident": r.meta["wrong_confident"],
                  "n_questions": r.meta["n_questions"], "rerun_from": r.meta["rerun_from"],
                  "questions": r.meta["questions"], "cost": r.meta["cost"],
                  "commitments": [[t.attribute, t.guess_value, t.probability] for t in r.turns]}
              for c, r in results[tid].items()} for tid in sorted(results)}, indent=1) + "\n")
    print(text)
    n_fail = sum(len(r.meta["parse_failed_turns"]) for v in results.values() for r in v.values())
    print(f"\n[parse] failed agent turns across reruns: {n_fail}")
    print(f"[cost] ${summ['cost']:.3f} in={summ['tokens_in']} out={summ['tokens_out']} "
          f"calls={summ['calls']} (estimate ${est:.2f}); runner {stats}")
    if not complete:
        print(f"[partial] not all units ran; rerun with --resume (partial: {rel(partial)})")
    write_run_json(out_dir / "summary.txt", {"script": "run_clarification_closed_loop", "args": vars(args),
                                             "model": spec.__dict__, "complete": complete,
                                             "conditions": conditions, "meter": summ, "runner": stats})
    lab = {c: np.mean([results[t][c].eval_label for t in test if c in results.get(t, {})])
           for c in conditions}
    print("[ledger] " + ledger_line(
        f"e8 closed-loop clarification (measured, {spec.short_name})",
        f"{len(test)} test sessions (split seed {args.split_seed}); conditions {', '.join(conditions)}",
        "hall. rate " + ", ".join(f"{c} {v:.3f}" for c, v in lab.items()), summ["cost"], rel(out_dir)))


if __name__ == "__main__":
    main()
