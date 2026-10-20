"""Bring cached healthcare-v2 resamples from k=3 to k=10 (PAID).

Same replay as eval/augment_healthcare_resamples.py: each turn that already
carries resamples (the UNDERDETERMINED attributes) is re-asked with its exact
original context (setup answers + prior-turn history rebuilt from the cache,
identical prompt template, system prompt, temperature 0.9), and k_target -
len(samples) NEW samples are appended. The existing 3 are kept first, unchanged
(meta["samples_original_k"] = 3). Guesses, signals and labels are untouched.
The input cache is only read; output goes to a NEW file.

--restrict test-union: only trajectories whose task_id is in the union of
the test splits of split seeds 42/123/456
(eval.score_trajectories.split_task_ids(ids, 0.7, seed)) are augmented;
others keep k=3 and are written unchanged. Cheaper, but the train splits then
mix k=3 and k=10, which matters for detectors that fit on samples (V4
posterior); zero-shot sampling baselines (SelfCheckGPT, semantic entropy)
are unaffected. Default: all.

A session counts as done only when every target turn has its new samples;
on a cost-cap stop, unfinished sessions keep k=3 (meta k10_status 'pending')
and --resume continues from the partial output.

Usage:
    python3 -m eval.paid.augment_resamples_k10 --model 70B --seed 42 --k-target 10 --max-cost 8
Writes data/trajectories/healthcare_v2_k10/<model>_<seed>_k10.jsonl (+ .run.json).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.paid.common import (
    ALLOWED_VALUES, PROTOCOLS, REPO, add_model_args, add_run_args, check_mock_out, ledger_line,
    make_client, preflight, print_estimate, protocol_of, rel, run_parallel, spec_from_args,
    write_run_json,
)
from eval.score_trajectories import split_task_ids
from pge.agents.paid_client import CostMeter, RoutedAgent
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2, USER_DEPENDENT_SCHEMA, dag_from_trajectory, sample_session_v2,
)
from pge.trajectory import load_jsonl, save_jsonl

IN_DIR = REPO / "data" / "trajectories" / "healthcare_v2"
OUT_DIR = REPO / "data" / "trajectories" / "healthcare_v2_k10"
SPLIT_SEEDS = (42, 123, 456)
# Ledger anchor: k=3 augmentation, 2,483 turns x 3 = 7,449 calls, $3.02 at $0.88/M
# -> ~460 tokens per call.
EST_IN, EST_OUT = 448, 12


TURN_SETS = ("sampled", "user")


def target_turns(traj, k_target, which="sampled"):
    """'sampled': turns that already carry resamples (the dev corpora sampled
    only the UNDERDETERMINED attributes). 'user': turns on schema
    user-dependent attributes (confirmatory corpora sample every turn)."""
    keep = (lambda t: bool(t.samples)) if which == "sampled" else \
        (lambda t: t.attribute in USER_DEPENDENT_SCHEMA)
    return [i for i, t in enumerate(traj.turns) if keep(t) and len(t.samples) < k_target]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="70B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=600, help="must match the generation run")
    ap.add_argument("--k-target", type=int, default=10)
    ap.add_argument("--restrict", choices=["all", "test-union", "test-42"], default="all",
                    help="test-42: only the test split of split seed 42 (cheapest)")
    ap.add_argument("--turns", choices=TURN_SETS, default=None,
                    help="which turns to augment (default: 'sampled' for v2 corpora, 'user' for v2fb)")
    ap.add_argument("--input", default=None, help="input cache (default data/trajectories/healthcare_v2/<model>_<seed>.jsonl)")
    add_run_args(ap, max_cost=8.0, concurrency=12)
    args = ap.parse_args(argv)
    check_mock_out(args)

    spec = spec_from_args(args)
    src = Path(args.input) if args.input else IN_DIR / f"{spec.short_name}_{args.seed}.jsonl"
    out = Path(args.out) if args.out else OUT_DIR / f"{spec.short_name}_{args.seed}_k{args.k_target}.jsonl"
    partial = Path(str(out) + ".partial.jsonl")
    if args.resume and partial.exists():
        trajs = load_jsonl(partial)
        print(f"[resume] from {rel(partial)}", flush=True)
    else:
        trajs = load_jsonl(src)
        for t in trajs:
            t.meta.setdefault("samples_original_k", max((len(x.samples) for x in t.turns), default=0))
    if any(t.model != spec.model_id for t in trajs):
        raise SystemExit(f"cache model {trajs[0].model} != --model {spec.model_id}: resamples must "
                         "come from the model that generated the cache")

    protocol = protocol_of(trajs)
    if args.turns is None:
        args.turns = "sampled" if protocol == "v2" else "user"
    eligible = {t.task_id for t in trajs}
    if args.restrict == "test-union":
        eligible = set()
        for s in SPLIT_SEEDS:
            eligible |= split_task_ids([t.task_id for t in trajs], 0.7, s)[1]
    elif args.restrict == "test-42":
        eligible = set(split_task_ids([t.task_id for t in trajs], 0.7, 42)[1])
    todo = [t for t in trajs if t.task_id in eligible and target_turns(t, args.k_target, args.turns)]
    n_calls = sum(args.k_target - len(t.turns[i].samples) for t in todo for i in target_turns(t, args.k_target, args.turns))
    n_turns = sum(len(target_turns(t, args.k_target, args.turns)) for t in todo)
    est = print_estimate(
        f"resamples k->{args.k_target} on {len(todo)} sessions / {n_turns} turns "
        f"(restrict={args.restrict}) {rel(src)} -> {rel(out)}",
        [("resample calls (T=0.9)", spec, n_calls, EST_IN, EST_OUT)], args.max_cost)
    if args.estimate_only:
        return

    rng = np.random.default_rng(args.seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(args.n)}
    for t in trajs:
        assert [x.attribute for x in t.turns] == sessions[t.task_id].schedule, \
            f"{t.task_id}: session stream mismatch"

    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "resample", args)
    preflight([client])

    def one(traj):
        session = sessions[traj.task_id]
        agent = RoutedAgent(
            attribute_schedule=[t.attribute for t in traj.turns], client=client,
            allowed_values=ALLOWED_VALUES, attribute_descriptions=ATTR_DESCRIPTIONS_V2,
            temperature=0.3, max_tokens=200, system_prompt=PROTOCOLS[protocol]["system_prompt"],
            feedback_display=PROTOCOLS[protocol]["feedback_display"], request_timeout=45.0,
        )
        full = dag_from_trajectory(traj)
        new = {}
        for i in target_turns(traj, args.k_target, args.turns):
            prefix = type(full)(nodes=full.nodes[:i], edges=[(p, c) for p, c in full.edges if c < i])
            need = args.k_target - len(traj.turns[i].samples)
            new[i] = agent.resample(i, session.setup_answers, prefix, k=need, temperature=0.9)
        return new, agent

    added = {"calls_short": 0}

    def on_result(traj, res):
        new, agent = res
        for i, s in new.items():
            traj.turns[i].samples = list(traj.turns[i].samples) + s
            added["calls_short"] += (args.k_target - len(traj.turns[i].samples))
        traj.meta["k10_status"] = "done"
        traj.meta["resample_k_target"] = args.k_target
        traj.meta["resample_aug_cost"] = round(agent.cost, 6)

    def checkpoint():
        save_jsonl(trajs, partial)

    for t in trajs:
        if t.task_id in eligible and target_turns(t, args.k_target, args.turns) and t.meta.get("k10_status") != "done":
            t.meta["k10_status"] = "pending"
    stats = run_parallel(todo, one, args.concurrency, meter, on_result, label="session",
                         progress_every=50, checkpoint=checkpoint)
    pending = [t.task_id for t in trajs if t.meta.get("k10_status") == "pending"]
    if pending:
        save_jsonl(trajs, partial)
        dest = partial
        print(f"[partial] {len(pending)} sessions still at k=3 -> {rel(partial)}; rerun with --resume",
              flush=True)
    else:
        save_jsonl(trajs, out)
        if partial.exists():
            partial.unlink()
        dest = out

    ks = [len(x.samples) for t in trajs for x in t.turns if x.samples]
    hist = {int(k): int(c) for k, c in zip(*np.unique(ks, return_counts=True))}
    summ = meter.summary()
    print(f"\n[done] {len(trajs)} trajectories -> {rel(dest)} ({stats['seconds']:.0f}s)")
    print(f"[samples] per-turn sample counts {hist}; failed resample calls (turn ended short of "
          f"k={args.k_target}): {added['calls_short']}")
    print(f"[cost] ${summ['cost']:.3f} in={summ['tokens_in']} out={summ['tokens_out']} "
          f"calls={summ['calls']} (estimate ${est:.2f})")
    write_run_json(out, {"script": "augment_resamples_k10", "args": vars(args), "model": spec.__dict__,
                         "source": rel(src), "sample_count_hist": hist, "pending": len(pending),
                         "meter": summ, "runner": stats})
    print("[ledger] " + ledger_line(
        f"healthcare resample augmentation k={args.k_target} ({spec.short_name}, seed {args.seed})",
        f"{n_turns} turns, +{n_calls} samples, restrict={args.restrict}; original 3 kept",
        f"sample counts {hist}", summ["cost"], rel(dest)))


if __name__ == "__main__":
    main()
