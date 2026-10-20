"""Healthcare-v2 generation with any registered model family (PAID).

Reuses eval.generate_healthcare_v2.run_llm_session unchanged (same session
stream sample_session_v2(default_rng(seed), i), task_id hcv2_<i>, same
per-session noise RNG default_rng((seed, i)), same eps channel, same DAG,
same prompts, same eval_label_v2). Only the agent is swapped for the
provider-routed, cost-capped RoutedAgent with a robust commitment parser.

Parse failures never crash the run. Per turn, meta["parse_status"] records
how the commitment was parsed (strict | lenient | lenient_no_attr |
retry1_* | strict_invalid_value | failed). A 'failed' turn keeps the legacy
placeholder (value 'unknown', p=0.5: below the 0.6 label threshold, so it
never enters the label's confident set) and is listed in
meta["parse_failed_turns"]; its feedback is the env's honest reaction to an
unusable commitment (truth-inconsistent, eps-flipped), exactly as in the
70B/8B corpora. 'strict_invalid_value' keeps the model's out-of-set value
(wrong vs truth by construction). Filter or report these in analysis.

Second domain (--domain travel_v1, protocol v2fb only): the same loop on the
travel-booking domain (pge/benchmarks/travel_v1.py): its session stream
sample_session(default_rng(seed), i), task_id tv1_<i>, truth, attributes,
descriptions, agent prompt (pge/prompts/travel_v1fb_system.txt) and labels.
The default --domain healthcare_v2 path is unchanged (same RNG streams,
outputs and trajectory meta).

Usage:
    python3 -m eval.paid.generate_healthcare_multimodel --model qwen72b --seed 42 --n 600 --max-cost 3
    python3 -m eval.paid.generate_healthcare_multimodel --model some/model --provider openrouter \
        --price-in 0.2 --price-out 0.2 --short-name mymodel
    python3 -m eval.paid.generate_healthcare_multimodel --domain travel_v1 --model gpt-4o-mini --n 600
Writes data/trajectories/healthcare_v2_models/<short_name>_<seed>.jsonl (protocol v2),
data/trajectories/healthcare_confirm/<short_name>_<seed>.jsonl (v2fb), or
data/trajectories/travel_confirm/<short_name>_<seed>.jsonl (travel_v1) (+ .run.json).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from eval.generate_healthcare_v2 import SYSTEM_PROMPT_HC, run_llm_session
from eval.paid.common import (
    ALLOWED_VALUES, REPO, add_model_args, add_run_args, check_mock_out, finalize,
    is_failed, ledger_line, load_partial, make_client, preflight, print_estimate, rel,
    run_parallel, save_partial, spec_from_args, write_run_json,
)
from eval.paid.common import ALLOWED_VALUES_BY_DOMAIN, PROTOCOLS, PROTOCOLS_BY_DOMAIN, add_dag_node
from pge.agents.paid_client import CostMeter, RoutedAgent
from pge.benchmarks import domains, healthcare_v2
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2, NO_FEEDBACK_ATTRS, AgentGuessV2, eval_label_user, eval_label_v2,
    sample_session_v2,
)
from pge.dag import ChainOfThoughtDAG
from pge.trajectory import Trajectory, Turn

OUT_DIRS = {"v2": REPO / "data" / "trajectories" / "healthcare_v2_models",
            "v2fb": REPO / "data" / "trajectories" / "healthcare_confirm"}
OUT_DIRS_BY_DOMAIN = {"healthcare_v2": OUT_DIRS,
                      "travel_v1": {"v2fb": REPO / "data" / "trajectories" / "travel_confirm"}}
DEFAULT_SEED = {"v2": 42, "v2fb": 7}
# Per agent call, from the 70B ledger: $2.46 / 5,909 turns at $0.88/M ~ 470 tokens.
# Resample calls: k=3 augmentation, 7,449 calls for $3.02 at $0.88/M ~ 460 tokens.
EST_IN, EST_OUT, EST_CALLS_PER_TURN = 470, 30, 1.1
EST_RS_IN, EST_RS_OUT = 448, 14
# Input-token scale of a domain's prompts relative to healthcare v2fb (system
# prompt + intake + history + description). Mock runs, n=120, seed 7: 476.6
# (travel) vs 487.4 (healthcare) chars/4 per call, i.e. travel is ~2% shorter;
# kept at 1.0 (conservative).
EST_IN_SCALE = {"healthcare_v2": 1.0, "travel_v1": 1.0}


def run_session_v2fb(session, agent: RoutedAgent, model_id: str, seed: int, epsilon: float,
                     rng: np.random.Generator, k: int, temperature: float, dom=None) -> Trajectory:
    """Confirmatory session loop. Same session, eps channel (same rng draws,
    in the same order) and DAG as eval/generate_healthcare_v2.run_llm_session;
    differences: the agent sees the raw reaction, and every turn gets k
    resamples from the exact context of its own guess (temperature T).
    `dom`: domain module from pge.benchmarks.domains (default healthcare_v2)."""
    dom = healthcare_v2 if dom is None else dom
    G = ChainOfThoughtDAG()
    node_idx: dict = {}
    turns: list[Turn] = []
    guesses: dict = {}
    for t, attribute in enumerate(session.schedule):
        guess, raw = agent.guess_full(t, session.setup_answers, G)
        samples = agent.resample(t, session.setup_answers, G, k=k, temperature=temperature) if k else []
        value = str(guess.value).strip().lower()
        guesses[attribute] = dom.AgentGuess(attribute, value, guess.probability)
        if attribute in dom.NO_FEEDBACK_ATTRS:
            signal = 0
        else:
            signal = 1 if value == session.user_state[attribute] else -1
            if rng.random() < epsilon:
                signal = -signal
        turns.append(Turn(attribute=attribute, guess_value=value, probability=guess.probability, turn=t,
                          raw_response=raw, samples=list(samples), signals={"noisy_feedback": signal}))
        add_dag_node(G, node_idx, attribute, value, guess.probability, signal, t, attributes=dom.ATTRIBUTES)
    meta = {"epsilon": epsilon, "n_turns": len(session.schedule),
            # second evaluation label (truth-derived, never detector-visible)
            "eval_label_user": dom.eval_label_user(guesses, session)}
    if dom is not healthcare_v2:  # healthcare trajectory meta stays exactly as before
        meta["domain"] = dom.DOMAIN
    return Trajectory(
        task_id=f"{dom.TASK_PREFIX}_{session.profile_id}", domain=dom.DOMAIN, model=model_id, seed=seed,
        turns=turns, eval_label=dom.eval_label(guesses, session), label_source="programmatic",
        meta=meta,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="qwen72b")
    ap.add_argument("--domain", choices=sorted(domains.DOMAINS), default="healthcare_v2",
                    help="private-attribute domain (travel_v1: protocol v2fb only)")
    ap.add_argument("--short-name", default=None, help="output file stem (default: registry short name)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v2fb",
                    help="v2fb (default): raw feedback shown to the agent + resamples on every turn; "
                         "v2: exact replay of the 2026-09 development protocol")
    ap.add_argument("--seed", type=int, default=None, help="default 7 for v2fb, 42 for v2")
    ap.add_argument("--resample-k", type=int, default=None, help="resamples per turn (default 3 for v2fb)")
    ap.add_argument("--resample-temperature", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--max-parse-fail", type=float, default=0.25,
                    help="stop early if > this fraction of turns fail to parse (after 10 sessions)")
    add_run_args(ap, max_cost=4.0)
    args = ap.parse_args(argv)
    check_mock_out(args)

    dom = domains.get(args.domain)
    hc = dom is healthcare_v2
    protocols = PROTOCOLS_BY_DOMAIN[args.domain]
    if args.protocol not in protocols:
        raise SystemExit(f"--domain {args.domain} supports --protocol {', '.join(sorted(protocols))} only")
    spec = spec_from_args(args, short_name=args.short_name)
    proto = protocols[args.protocol]
    if args.seed is None:
        args.seed = DEFAULT_SEED[args.protocol]
    if args.resample_k is None:
        args.resample_k = 3 if args.protocol == "v2fb" else 0
    if args.protocol == "v2" and args.resample_k:
        raise SystemExit("--protocol v2 replays the legacy loop, which has no resampling")
    out_dirs = OUT_DIRS_BY_DOMAIN[args.domain]
    out = Path(args.out) if args.out else out_dirs[args.protocol] / f"{spec.short_name}_{args.seed}.jsonl"
    rng = np.random.default_rng(args.seed)
    sessions = [dom.sample_session(rng, i) for i in range(args.n)]
    noise_rngs = [np.random.default_rng((args.seed, i)) for i in range(args.n)]
    n_turns = sum(len(s.schedule) for s in sessions)

    scale = EST_IN_SCALE[args.domain]
    est_in, est_rs_in = (EST_IN, EST_RS_IN) if hc else (round(EST_IN * scale), round(EST_RS_IN * scale))
    rows = [("agent turns", spec, int(n_turns * EST_CALLS_PER_TURN), est_in, EST_OUT)]
    if args.resample_k:
        rows.append((f"resamples (k={args.resample_k}, T={args.resample_temperature})", spec,
                     n_turns * args.resample_k, est_rs_in, EST_RS_OUT))
    est = print_estimate(
        f"{'healthcare' if hc else args.domain} {args.protocol} n={args.n} seed {args.seed} "
        f"({n_turns} turns) -> {rel(out)}",
        rows, args.max_cost,
    )
    if args.estimate_only:
        return

    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "agent", args)
    preflight([client])

    done = {t.task_id: t for t in load_partial(out)} if args.resume else {}
    if done:
        print(f"[resume] {len(done)} sessions already in {rel(out)}.partial.jsonl", flush=True)
    todo = [i for i in range(args.n) if f"{dom.TASK_PREFIX}_{i}" not in done]
    statuses: Counter = Counter(s for t in done.values() for s in t.meta.get("parse_status", []))

    def one(i: int):
        agent = RoutedAgent(
            attribute_schedule=sessions[i].schedule,
            client=client,
            allowed_values=ALLOWED_VALUES_BY_DOMAIN[args.domain],
            attribute_descriptions=dom.ATTR_DESCRIPTIONS,
            temperature=0.3,
            max_tokens=200,
            system_prompt=proto["system_prompt"],
            feedback_display=proto["feedback_display"],
            reactor=dom.USER_ROLE,
            request_timeout=45.0,
        )
        if args.protocol == "v2":
            traj, agent = run_llm_session(sessions[i], spec.model_id, args.seed, args.epsilon,
                                          noise_rngs[i], agent=agent)
        else:
            traj = run_session_v2fb(sessions[i], agent, spec.model_id, args.seed, args.epsilon,
                                    noise_rngs[i], args.resample_k, args.resample_temperature, dom=dom)
        assert len(agent.parse_log) == len(traj.turns)
        traj.meta.update({
            "protocol": args.protocol,
            "feedback_display": proto["feedback_display"],
            "resample_k": args.resample_k,
            "resample_temperature": args.resample_temperature if args.resample_k else None,
            "provider": spec.provider,
            "short_name": spec.short_name,
            "parse_status": list(agent.parse_log),
            "parse_failed_turns": [k for k, s in enumerate(agent.parse_log) if is_failed(s)],
            "tokens_in": agent.input_tokens_used,
            "tokens_out": agent.output_tokens_used,
            "cost": round(agent.cost, 6),
        })
        return traj

    def on_result(i, traj):
        done[traj.task_id] = traj
        statuses.update(traj.meta["parse_status"])
        total = sum(statuses.values())
        if len(done) >= 10 and total and statuses["failed"] / total > args.max_parse_fail:
            if not meter.tripped:
                print(f"[abort] parse-failure rate {statuses['failed']/total:.1%} > "
                      f"{args.max_parse_fail:.0%}: model does not follow the format", flush=True)
            meter.stop(f"parse-failure rate > {args.max_parse_fail:.0%}")

    stats = run_parallel(todo, one, args.concurrency, meter, on_result, label="session",
                         checkpoint=lambda: save_partial(list(done.values()), out))
    if not done:
        raise SystemExit("no sessions completed")
    results = list(done.values())
    complete = len(results) == args.n
    if complete:
        n_written = finalize(results, out)
    else:
        save_partial(results, out)
        n_written = len(results)
        print(f"[partial] {n_written}/{args.n} sessions saved to {rel(out)}.partial.jsonl; "
              "rerun with --resume (and a higher --max-cost if the cap stopped it)", flush=True)

    labels = np.array([t.eval_label for t in results])
    total = sum(statuses.values())
    n_failed_sessions = sum(1 for t in results if t.meta["parse_failed_turns"])
    summ = meter.summary()
    print(f"\n[done] {n_written} trajectories -> {rel(out) if complete else rel(out) + '.partial.jsonl'}"
          f"  ({stats['seconds']:.0f}s)")
    print(f"[labels] hallucinating {labels.sum()}/{len(labels)} = {labels.mean():.1%}")
    print(f"[parse] {dict(statuses)}  failed turns {statuses['failed']}/{total} = "
          f"{statuses['failed']/max(total,1):.2%}; sessions with >=1 failed turn: {n_failed_sessions}")
    print(f"[cost] ${summ['cost']:.3f}  in={summ['tokens_in']} out={summ['tokens_out']} "
          f"calls={summ['calls']}  (${summ['cost']/len(results):.4f}/session; estimate was ${est:.2f})")
    run = {"script": "generate_healthcare_multimodel", "domain": args.domain, "args": vars(args),
           "model": spec.__dict__, "complete": complete, "n_written": n_written,
           "parse_status_counts": dict(statuses), "hall_rate": float(labels.mean()), "meter": summ,
           "runner": stats}
    if not hc:
        run["hall_rate_user"] = float(np.mean([t.meta["eval_label_user"] for t in results]))
        print(f"[labels] user-specific L_U {run['hall_rate_user']:.1%}")
    write_run_json(out, run)
    print("[ledger] " + ledger_line(
        f"{'healthcare v2' if hc else args.domain} corpus ({spec.short_name}, seed {args.seed}, n={n_written})",
        f"eps={args.epsilon} noisy partial feedback, {spec.model_id} via {spec.provider}; "
        f"parse-failed turns {statuses['failed']}/{total}",
        f"{n_written} trajectories, {labels.mean():.1%} hallucinating", summ["cost"],
        rel(out) + ("" if complete else ".partial.jsonl")))


if __name__ == "__main__":
    main()
