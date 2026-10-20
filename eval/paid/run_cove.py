"""Chain-of-Verification baseline on healthcare-v2 test splits (PAID).

Runs pge.baselines.cove.ChainOfVerification unchanged, with an llm_fn routed
to the chosen provider/model (default: the agent's own model, 70B, T=0, as in
self-verification). Healthcare caches store no per-turn prompt, so the
module would fall back to "(no additional context)". This script fills
turn.prompt on a COPY of each trajectory with the turn's ORIGINAL decision
context, rebuilt from detector-visible inputs only: the setup (intake)
answers, the attribute description, and the agent's prior commitments with
their feedback marks (the same history window the agent saw; the turn's own
answer is excluded). No ground truth and no labels enter any prompt.

CoVe is zero-shot, so each task is scored once and reused across the
split seeds 42/123/456 (test sets via split_task_ids(ids, 0.7, seed); union
of test task ids is scored). Every response is cached in
<out>/cove_cache.jsonl keyed by (task_id, sha256(model + prompt)); reruns
hit the cache for free.

Usage:
    python3 -m eval.paid.run_cove --max-cost 3
    python3 -m eval.paid.run_cove --model gpt-4o-mini --max-cost 1
Writes results/cove_baseline/<verifier>_on_<corpus>/{scores.json, seed<k>.npz, cove_inputs.jsonl, cove_cache.jsonl}.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from eval.paid.common import (
    REPO, add_model_args, add_run_args, check_mock_out, ledger_line, make_client, preflight,
    print_estimate, rel, run_parallel, spec_from_args, write_run_json,
)
from eval.score_trajectories import average_precision, split_task_ids
from pge.agents.llama_together import _format_history
from pge.baselines.cove import VERIFY_PROMPT, ChainOfVerification
from pge.baselines.semantic_entropy import _extract_final_value
from pge.agents.paid_client import CostMeter
from pge.benchmarks.healthcare_v2 import ATTR_DESCRIPTIONS_V2, dag_from_trajectory
from pge.trajectory import load_jsonl, save_jsonl

SPLIT_SEEDS = (42, 123, 456)
VERIFY_TEMP, VERIFY_MAX_TOKENS = 0.0, 40


def turn_context(traj, t: int, setup: dict) -> str:
    """Detector-visible decision context of turn t (no truth, no labels)."""
    full = dag_from_trajectory(traj)
    prefix = type(full)(nodes=full.nodes[:t], edges=[(p, c) for p, c in full.edges if c < t])
    a = traj.turns[t].attribute
    setup_str = "\n".join(f"  {k}: {v}" for k, v in setup.items())
    return (f"Patient intake answers:\n{setup_str}\n\n"
            f"Earlier commitments in this session (feedback from the patient's behavior):\n"
            f"{_format_history(prefix)}\n\n"
            f"Attribute definition: {ATTR_DESCRIPTIONS_V2.get(a, '')}")


def with_contexts(traj, setup: dict):
    turns = [dataclasses.replace(tt, prompt=turn_context(traj, i, setup)) for i, tt in enumerate(traj.turns)]
    return dataclasses.replace(traj, turns=turns, eval_label=0)  # label masked for the detector


class ResponseCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[(r["task_id"], r["key"])] = r["response"]

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:32]

    def get(self, tid, key):
        return self.data.get((tid, key))

    def put(self, rec: dict) -> None:
        with self.lock:
            self.data[(rec["task_id"], rec["key"])] = rec["response"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="70B", help_role="verifier")
    ap.add_argument("--corpus", default="data/trajectories/healthcare_v2/70B_42.jsonl")
    ap.add_argument("--gen-seed", type=int, default=42, help="generation seed of the corpus")
    ap.add_argument("--n", type=int, default=600, help="must match the generation run")
    add_run_args(ap, max_cost=3.0)
    args = ap.parse_args(argv)
    check_mock_out(args)

    from pge.benchmarks.healthcare_v2 import sample_session_v2

    spec = spec_from_args(args)
    corpus = Path(args.corpus) if Path(args.corpus).is_absolute() else REPO / args.corpus
    out_dir = Path(args.out) if args.out else \
        REPO / "results" / "cove_baseline" / f"{spec.short_name}_on_{corpus.stem}"
    trajs = load_jsonl(corpus)
    by_id = {t.task_id: t for t in trajs}
    rng = np.random.default_rng(args.gen_seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(args.n)}

    tests = {s: split_task_ids(list(by_id), 0.7, s)[1] for s in SPLIT_SEEDS}
    union = sorted(set().union(*tests.values()), key=lambda s: int(s.split("_")[1]))
    inputs = {}
    for tid in union:
        assert [x.attribute for x in by_id[tid].turns] == sessions[tid].schedule, f"{tid}: stream mismatch"
        inputs[tid] = with_contexts(by_id[tid], sessions[tid].setup_answers)

    cache = ResponseCache(out_dir / "cove_cache.jsonl")
    prompts = {tid: [VERIFY_PROMPT.format(context=t.prompt, attribute=t.attribute) for t in tr.turns]
               for tid, tr in inputs.items()}
    uncached = [(tid, p) for tid, ps in prompts.items() for p in ps
                if cache.get(tid, ResponseCache.key(spec.model_id, p)) is None]
    avg_chars = np.mean([len(p) for _, p in uncached]) if uncached else 0
    est = print_estimate(
        f"CoVe on {len(union)} test-union tasks ({sum(len(v) for v in prompts.values())} turns, "
        f"{len(uncached)} uncached) of {rel(corpus)}, verifier {spec.model_id} -> {rel(out_dir)}",
        [("verification queries", spec, len(uncached), avg_chars / 3.6 + 20, 15)], args.max_cost)
    if args.estimate_only:
        return
    save_jsonl(list(inputs.values()), out_dir / "cove_inputs.jsonl")  # lint + audit of detector inputs

    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "verifier", args)
    if uncached:
        preflight([client])
    hits = {"cache": 0, "api": 0}

    def score_task(tid):
        def llm_fn(prompt: str) -> str:
            k = ResponseCache.key(spec.model_id, prompt)
            r = cache.get(tid, k)
            if r is not None:
                hits["cache"] += 1
                return r
            text, tin, tout, cost = client.chat_full([{"role": "user", "content": prompt}],
                                                     VERIFY_TEMP, VERIFY_MAX_TOKENS)
            hits["api"] += 1
            cache.put({"task_id": tid, "key": k, "model": spec.model_id, "provider": spec.provider,
                       "response": text, "tokens_in": tin, "tokens_out": tout, "cost": cost})
            return text
        det = ChainOfVerification(llm_fn)
        return det.score(inputs[tid])

    scores: dict = {}
    stats = run_parallel(union, score_task, args.concurrency, meter,
                         lambda tid, s: scores.__setitem__(tid, s), label="task", progress_every=50)

    # parse audit (cove.py counts an unparseable re-answer as disagreement)
    unparsed = sum(1 for tid in scores for p in prompts[tid]
                   if not _extract_final_value(cache.get(tid, ResponseCache.key(spec.model_id, p)) or ""))
    n_q = sum(len(prompts[tid]) for tid in scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed, lines = {}, []
    for s in SPLIT_SEEDS:
        ids = sorted(tests[s], key=lambda x: int(x.split("_")[1]))
        done = [t for t in ids if t in scores]
        y = np.array([by_id[t].eval_label for t in done])
        sc = np.array([scores[t] for t in done])
        a, ap_ = (auroc(sc, y), average_precision(sc, y)) if len(done) else (float("nan"),) * 2
        per_seed[str(s)] = {"task_ids": done, "scores": sc.tolist(), "complete": len(done) == len(ids),
                            "auroc": a, "auprc": ap_}
        np.savez(out_dir / f"seed{s}.npz", task_ids=np.array(done, dtype=str), score_cove=sc, labels=y)
        lines.append(f"  seed {s}: cove AUROC {a:.3f}  AUPRC {ap_:.3f}  (n={len(done)}/{len(ids)})")
    summ = meter.summary()
    # scores.json only when every split is complete (run_paid.sh treats it as
    # the done marker); otherwise scores.partial.json and a rerun resumes from
    # the response cache.
    complete = all(per_seed[str(s)]["complete"] for s in SPLIT_SEEDS)
    scores_path = out_dir / ("scores.json" if complete else "scores.partial.json")
    if complete:
        for stale in ("scores.partial.json", "scores.partial.json.run.json"):
            if (out_dir / stale).exists():
                (out_dir / stale).unlink()
    scores_path.write_text(json.dumps({
        "detector": "cove", "verifier": spec.model_id, "provider": spec.provider, "corpus": rel(corpus),
        "context": "setup answers + attribute definition + prior commitments with feedback marks",
        "temperature": VERIFY_TEMP, "score_by_task": scores, "per_seed": per_seed,
        "unparsed_replies": unparsed, "n_queries": n_q, "cost": summ["cost"]}, indent=1) + "\n")
    print("\n".join(lines))
    m = np.nanmean([per_seed[str(s)]["auroc"] for s in SPLIT_SEEDS])
    print(f"  mean AUROC {m:.3f}")
    print(f"[parse] replies without a [final_answer] line (scored as disagreement by cove.py): "
          f"{unparsed}/{n_q}")
    print(f"[cost] ${summ['cost']:.3f} in={summ['tokens_in']} out={summ['tokens_out']} api calls="
          f"{hits['api']} cache hits={hits['cache']} (estimate ${est:.2f}); runner {stats}")
    write_run_json(scores_path, {"script": "run_cove", "complete": complete, "args": vars(args), "model": spec.__dict__,
                                             "meter": summ, "runner": stats, "cache_hits": hits})
    print("[ledger] " + ledger_line(
        f"CoVe baseline ({spec.short_name} verifier) on {corpus.stem}",
        f"{len(scores)}/{len(union)} test-union tasks, split seeds {SPLIT_SEEDS}, T=0, context rebuilt "
        f"from detector-visible inputs; unparsed {unparsed}/{n_q}",
        f"cove mean AUROC {m:.3f} (" + ", ".join(f"{per_seed[str(s)]['auroc']:.3f}" for s in SPLIT_SEEDS) + ")",
        summ["cost"], rel(out_dir)))


if __name__ == "__main__":
    main()
