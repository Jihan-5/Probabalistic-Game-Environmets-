"""LLM-judge baseline (PAID, zero-shot): a second model reads the whole session.

Review asked for the obvious agentic baseline: a separate LLM that reads the
transcript, including the user's reactions, and judges which commitments are
wrong. It is a feedback detector (not bounded by Theorem 1). For each session
the judge sees only detector-visible inputs: the intake answers, every
attribute's definition, and each turn's commitment (value, confidence) with
the user's reaction exactly as the agent saw it in protocol v2fb ("+", "-",
"none"). It returns, per commitment, the probability that the commitment is
wrong for this user. The session score is the probability of the label event
under those per-commitment probabilities treated as independent (the same
Poisson-binomial tail over confident commitments on the schema-level
user-dependent attributes that V6/V7 use for L_U); the plain mean of the
probabilities is stored as well. No truth and no labels enter any prompt.

Zero-shot, so each test-union session (split seeds 42/123/456) is judged once.
Responses are cached; reruns are free. Cost: ~400 sessions per corpus x
~1.3k input tokens; GPT-4o-mini via OpenRouter is about $0.15 per corpus.

    python3 -m eval.paid.run_llm_judge --corpus data/trajectories/healthcare_confirm/8B_7.jsonl --max-cost 0.5
    python3 -m eval.paid.run_llm_judge --mock --out /tmp/judge_mock --corpus ...

Writes results/llm_judge/<judge>_on_<corpus stem>/{scores.json, judge_cache.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.bootstrap_ci import auroc
from eval.paid.common import (
    REPO, add_model_args, add_run_args, check_mock_out, ledger_line, make_client, preflight, print_estimate,
    rel, run_parallel, spec_from_args, write_run_json,
)
from eval.paid.run_cove import ResponseCache
from eval.score_trajectories import split_task_ids
from pge.agents.paid_client import CostMeter
from pge.baselines.posterior import label_threshold, poisson_binomial_tail, turn_signal
from pge.benchmarks import healthcare_v2 as dom
from pge.trajectory import load_jsonl

SPLIT_SEEDS = (42, 123, 456)
TEMP, MAX_TOKENS = 0.0, 300

PROMPT = """You are auditing an AI scheduling assistant at an imaging clinic. For one patient, the assistant
stated a series of assumptions ("commitments"), each with its own confidence. After each one, the patient's
behaviour gave an implicit reaction: "+" (seemed to accept it), "-" (seemed to contradict it) or "none".
Reactions are informative but noisy: patients sometimes react wrongly, and some attributes never get a reaction.
The assistant only knew the patient's intake answers below; it was never told the patient's private details.

Patient intake answers:
{intake}

Attribute definitions:
{defs}

Session transcript (turn: attribute = committed value, assistant confidence, patient reaction):
{turns}

For EACH turn, estimate the probability (0 to 1) that the committed value is WRONG for this patient.
Answer with JSON only, in turn order, exactly {n} numbers: {{"p_wrong": [p1, p2, ...]}}"""


def session_prompt(traj, setup):
    sym = {1: "+", -1: "-", 0: "none"}
    turns = "\n".join(f"{i + 1}: {t.attribute} = {t.guess_value}, confidence {t.probability:.2f}, "
                      f"reaction {sym[turn_signal(t)]}" for i, t in enumerate(traj.turns))
    attrs = sorted({t.attribute for t in traj.turns})
    defs = "\n".join(f"- {a}: {dom.ATTR_DESCRIPTIONS_V2.get(a, '')}" for a in attrs)
    intake = "\n".join(f"- {k}: {v}" for k, v in setup.items())
    return PROMPT.format(intake=intake, defs=defs, turns=turns, n=len(traj.turns))


def parse(text, n):
    m = re.search(r"\[([^\[\]]*)\]", text or "")
    if not m:
        return None
    try:
        vals = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
    except ValueError:
        return None
    vals = [min(max(v / 100 if v > 1 else v, 0.0), 1.0) for v in vals]
    return vals if len(vals) == n else None


def session_score(traj, p):
    q = [pi for t, pi in zip(traj.turns, p) if t.probability >= 0.6 and t.attribute in dom.USER_DEPENDENT_SCHEMA]
    tail = poisson_binomial_tail(np.array(q), label_threshold(len(q), 0.5)) if q else 0.0
    return float(tail), float(np.mean(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="gpt-4o-mini", help_role="judge")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gen-seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=600)
    add_run_args(ap, max_cost=0.5)
    args = ap.parse_args(argv)
    check_mock_out(args)
    spec = spec_from_args(args)
    corpus = Path(args.corpus) if Path(args.corpus).is_absolute() else REPO / args.corpus
    out_dir = Path(args.out) if args.out else REPO / "results" / "llm_judge" / f"{spec.short_name}_on_{corpus.stem}"
    trajs = load_jsonl(corpus)
    by_id = {t.task_id: t for t in trajs}
    rng = np.random.default_rng(args.gen_seed)
    sessions = {f"hcv2_{i}": dom.sample_session_v2(rng, i) for i in range(args.n)}
    tests = {s: split_task_ids(list(by_id), 0.7, s)[1] for s in SPLIT_SEEDS}
    union = sorted(set().union(*tests.values()), key=lambda s: int(s.split("_")[1]))
    prompts = {tid: session_prompt(by_id[tid], sessions[tid].setup_answers) for tid in union}
    cache = ResponseCache(out_dir / "judge_cache.jsonl")
    uncached = [tid for tid in union if cache.get(tid, ResponseCache.key(spec.model_id, prompts[tid])) is None]
    avg = np.mean([len(prompts[t]) for t in uncached]) if uncached else 0
    est = print_estimate(f"LLM judge on {len(union)} test-union sessions of {rel(corpus)} ({len(uncached)} uncached)",
                         [("judge calls", spec, len(uncached), avg / 3.6, 90)], args.max_cost)
    if args.estimate_only:
        return
    meter = CostMeter(args.max_cost)
    client = make_client(spec, meter, "judge", args)
    if uncached:
        preflight([client])
    out = {}

    def judge(tid):
        k = ResponseCache.key(spec.model_id, prompts[tid])
        text = cache.get(tid, k)
        if text is None:
            text, tin, tout, cost = client.chat_full([{"role": "user", "content": prompts[tid]}], TEMP, MAX_TOKENS)
            cache.put({"task_id": tid, "key": k, "model": spec.model_id, "provider": spec.provider,
                       "response": text, "tokens_in": tin, "tokens_out": tout, "cost": cost})
        return text

    stats = run_parallel(union, judge, args.concurrency, meter, lambda tid, txt: out.__setitem__(tid, txt),
                         label="session", progress_every=50)
    probs, failed = {}, 0
    for tid, txt in out.items():
        p = parse(txt, len(by_id[tid].turns))
        if p is None:
            failed += 1
            p = [0.5] * len(by_id[tid].turns)  # unparseable: uninformative
        probs[tid] = p
    scores = {tid: session_score(by_id[tid], p)[0] for tid, p in probs.items()}
    means = {tid: session_score(by_id[tid], p)[1] for tid, p in probs.items()}
    complete = all(t in scores for t in union)
    summ = meter.summary()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("scores.json" if complete else "scores.partial.json")
    path.write_text(json.dumps({"detector": "llm_judge", "judge": spec.model_id, "provider": spec.provider,
                                "corpus": rel(corpus), "score_by_task": scores, "mean_by_task": means,
                                "p_wrong_by_task": probs, "unparsed": failed, "n": len(out),
                                "complete": complete, "cost": summ["cost"]}, indent=1) + "\n")
    # quick look on the stored eval_label (composite L); L_U is computed in the analysis
    for s in SPLIT_SEEDS:
        ids = [t for t in tests[s] if t in scores]
        y = np.array([by_id[t].eval_label for t in ids])
        print(f"  seed {s}: AUROC on L (composite) {auroc(np.array([scores[t] for t in ids]), y):.3f} (n={len(ids)})")
    print(f"[parse] unparseable judge replies: {failed}/{len(out)}")
    print(f"[cost] ${summ['cost']:.3f} in={summ['tokens_in']} out={summ['tokens_out']} (estimate ${est:.2f}); runner {stats}")
    write_run_json(path, {"script": "run_llm_judge", "complete": complete, "args": vars(args), "model": spec.__dict__,
                          "meter": summ, "runner": stats})
    print("[ledger] " + ledger_line(f"LLM-judge baseline ({spec.short_name}) on {corpus.stem}",
                                    f"{len(out)}/{len(union)} test-union sessions, zero-shot, T=0; unparsed {failed}",
                                    "see analysis", summ["cost"], rel(out_dir)))


if __name__ == "__main__":
    main()
