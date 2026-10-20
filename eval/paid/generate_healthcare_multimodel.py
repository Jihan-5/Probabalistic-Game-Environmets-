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

Usage:
    python3 -m eval.paid.generate_healthcare_multimodel --model qwen72b --seed 42 --n 600 --max-cost 3
    python3 -m eval.paid.generate_healthcare_multimodel --model some/model --provider openrouter \
        --price-in 0.2 --price-out 0.2 --short-name mymodel
Writes data/trajectories/healthcare_v2_models/<short_name>_<seed>.jsonl (+ .run.json).
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
from pge.agents.paid_client import CostMeter, RoutedAgent
from pge.benchmarks.healthcare_v2 import ATTR_DESCRIPTIONS_V2, sample_session_v2

OUT_DIR = REPO / "data" / "trajectories" / "healthcare_v2_models"
# Per agent call, from the 70B ledger: $2.46 / 5,909 turns at $0.88/M ~ 470 tokens.
EST_IN, EST_OUT, EST_CALLS_PER_TURN = 470, 30, 1.1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="qwen72b")
    ap.add_argument("--short-name", default=None, help="output file stem (default: registry short name)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--max-parse-fail", type=float, default=0.25,
                    help="stop early if > this fraction of turns fail to parse (after 10 sessions)")
    add_run_args(ap, max_cost=4.0)
    args = ap.parse_args(argv)
    check_mock_out(args)

    spec = spec_from_args(args, short_name=args.short_name)
    out = Path(args.out) if args.out else OUT_DIR / f"{spec.short_name}_{args.seed}.jsonl"
    rng = np.random.default_rng(args.seed)
    sessions = [sample_session_v2(rng, i) for i in range(args.n)]
    noise_rngs = [np.random.default_rng((args.seed, i)) for i in range(args.n)]
    n_turns = sum(len(s.schedule) for s in sessions)

    est = print_estimate(
        f"healthcare-v2 multimodel n={args.n} ({n_turns} turns) -> {rel(out)}",
        [("agent turns", spec, int(n_turns * EST_CALLS_PER_TURN), EST_IN, EST_OUT)],
        args.max_cost,
    )
    if args.estimate_only:
        return

    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "agent", args)
    preflight([client])

    done = {t.task_id: t for t in load_partial(out)} if args.resume else {}
    if done:
        print(f"[resume] {len(done)} sessions already in {rel(out)}.partial.jsonl", flush=True)
    todo = [i for i in range(args.n) if f"hcv2_{i}" not in done]
    statuses: Counter = Counter(s for t in done.values() for s in t.meta.get("parse_status", []))

    def one(i: int):
        agent = RoutedAgent(
            attribute_schedule=sessions[i].schedule,
            client=client,
            allowed_values=ALLOWED_VALUES,
            attribute_descriptions=ATTR_DESCRIPTIONS_V2,
            temperature=0.3,
            max_tokens=200,
            system_prompt=SYSTEM_PROMPT_HC,
            request_timeout=45.0,
        )
        traj, agent = run_llm_session(sessions[i], spec.model_id, args.seed, args.epsilon,
                                      noise_rngs[i], agent=agent)
        assert len(agent.parse_log) == len(traj.turns)
        traj.meta.update({
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
    run = {"script": "generate_healthcare_multimodel", "args": vars(args), "model": spec.__dict__,
           "complete": complete, "n_written": n_written, "parse_status_counts": dict(statuses),
           "hall_rate": float(labels.mean()), "meter": summ, "runner": stats}
    write_run_json(out, run)
    print("[ledger] " + ledger_line(
        f"healthcare v2 corpus ({spec.short_name}, seed {args.seed}, n={n_written})",
        f"eps={args.epsilon} noisy partial feedback, {spec.model_id} via {spec.provider}; "
        f"parse-failed turns {statuses['failed']}/{total}",
        f"{n_written} trajectories, {labels.mean():.1%} hallucinating", summ["cost"],
        rel(out) + ("" if complete else ".partial.jsonl")))


if __name__ == "__main__":
    main()
