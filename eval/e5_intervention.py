"""e5_intervention — measured mitigation via revisitation (healthcare v2).

Measured mitigation rate: flag
high-risk test sessions with a detector trained on the train split, re-run
ONLY the flagged sessions with a revisitation prompt (the agent sees its own
prior commitments and which drew negative feedback, and is instructed to
re-derive with calibrated caution), and measure how many flagged
hallucinating sessions become clean — plus the collateral rate on falsely
flagged clean sessions.

Sessions are regenerated deterministically (same seed stream as
eval/generate_healthcare_v2.py), so the re-run faces the same patient.

Usage:
    python3 -m eval.e5_intervention --n 600 --model 70B --seed 42 \
        [--flag-frac 0.3] [--max-cost 2.0]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pge.agents.llama_together import (
    MODEL_LLAMA_8B,
    MODEL_LLAMA_70B,
    PRICE_PER_M,
    LlamaTogetherAgent,
)
from pge.baselines.dag_logreg import DagFeatureClassifier
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2,
    ATTRIBUTES_V2,
    NO_FEEDBACK_ATTRS,
    AgentGuessV2,
    dag_from_trajectory,
    eval_label_v2,
    sample_session_v2,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score
from pge.trajectory import Trajectory, load_jsonl

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "e5_intervention"
SYSTEM_PROMPT_HC = (REPO / "pge" / "prompts" / "healthcare_v2_system.txt").read_text()
REVISIT_PREAMBLE = (
    "REVISITATION: In an earlier pass on this same patient, you made the "
    "following commitments. Feedback '-' means the patient's behavior "
    "contradicted the commitment; '+' means it was consistent; '0' means no "
    "feedback.\n{history}\n"
    "Several confident commitments were contradicted. Re-derive each "
    "attribute from the intake answers alone. Where the intake genuinely "
    "does not determine the attribute, output p=0.5 or lower. Do NOT repeat "
    "a contradicted commitment at high confidence.\n\n"
)


def _prior_history_block(traj: Trajectory) -> str:
    lines = []
    for t in traj.turns:
        sig = t.signals.get("noisy_feedback", 0)
        mark = "+" if sig > 0 else "-" if sig < 0 else "0"
        lines.append(f"  [{t.attribute}]: {t.guess_value}, p={t.probability:.2f}  ({mark})")
    return "\n".join(lines)


def rerun_with_revisitation(
    session, prior: Trajectory, model: str, epsilon: float, rng: np.random.Generator
):
    agent = LlamaTogetherAgent(
        attribute_schedule=session.schedule,
        model=model,
        attribute_descriptions=ATTR_DESCRIPTIONS_V2,
        temperature=0.3,
        max_tokens=200,
        system_prompt=SYSTEM_PROMPT_HC,
        request_timeout=45.0,
    )
    preamble = REVISIT_PREAMBLE.format(history=_prior_history_block(prior))
    setup = dict(session.setup_answers)
    setup["revisitation_notes"] = preamble

    G = ChainOfThoughtDAG()
    node_idx: dict[str, int] = {}
    guesses: dict[str, AgentGuessV2] = {}
    for t, attribute in enumerate(session.schedule):
        guess, _ = agent.guess_full(t, setup, G)
        value = str(guess.value).strip().lower()
        guesses[attribute] = AgentGuessV2(attribute, value, guess.probability)
        correct = value == session.user_state[attribute]
        signal = 0 if attribute in NO_FEEDBACK_ATTRS else (1 if correct else -1)
        if signal != 0 and rng.random() < epsilon:
            signal = -signal
        idx = G.add_node(
            GuessNode(
                attribute=attribute, value=value,
                probability=guess.probability,
                score=bayesian_score(guess.probability, signal), turn=t,
            )
        )
        node_idx[attribute] = idx
        for dep in ATTRIBUTES_V2[attribute][1]:
            if dep.startswith("attr:") and dep[5:] in node_idx and node_idx[dep[5:]] < idx:
                G.add_edge(node_idx[dep[5:]], idx)
        if idx > 0 and (idx - 1, idx) not in G.edges:
            G.add_edge(idx - 1, idx)
    return eval_label_v2(guesses, session), agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="must match the generation run")
    ap.add_argument("--model", default="70B", choices=["8B", "70B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--flag-frac", type=float, default=0.3)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=2.0)
    args = ap.parse_args()

    model = MODEL_LLAMA_8B if args.model == "8B" else MODEL_LLAMA_70B
    in_p, out_p = PRICE_PER_M[model]

    cache = REPO / "data" / "trajectories" / "healthcare_v2" / f"{args.model}_{args.seed}.jsonl"
    trajs = load_jsonl(cache)
    by_id = {t.task_id: t for t in trajs}

    # Regenerate the identical session stream.
    rng = np.random.default_rng(args.seed)
    sessions = {f"hcv2_{i}": sample_session_v2(rng, i) for i in range(args.n)}

    ids = sorted(by_id)
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(len(ids))
    n_train = int(args.train_frac * len(ids))
    train_ids = {ids[i] for i in order[:n_train]}
    test_ids = [ids[i] for i in order[n_train:]]

    dags = {tid: dag_from_trajectory(by_id[tid]) for tid in ids}
    clf = DagFeatureClassifier().fit(
        [by_id[t] for t in train_ids],
        [dags[t] for t in train_ids],
        [by_id[t].eval_label for t in train_ids],
    )
    risk = {t: clf.score(by_id[t], dags[t]) for t in test_ids}
    n_flag = max(1, int(args.flag_frac * len(test_ids)))
    flagged = sorted(test_ids, key=lambda t: -risk[t])[:n_flag]

    labels_before = {t: by_id[t].eval_label for t in flagged}
    print(f"[flag] {len(flagged)}/{len(test_ids)} test sessions flagged "
          f"({sum(labels_before.values())} actually hallucinating)", flush=True)

    tokens = [0, 0]
    results = {}
    noise_rngs = {
        t: np.random.default_rng((args.seed, 7919, int(t.split("_")[1])))
        for t in flagged
    }

    def one(tid):
        return tid, rerun_with_revisitation(
            sessions[tid], by_id[tid], model, args.epsilon, noise_rngs[tid]
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(one, tid) for tid in flagged]
        for fut in as_completed(futures):
            try:
                tid, (label_after, agent) = fut.result()
                results[tid] = label_after
                tokens[0] += agent.input_tokens_used
                tokens[1] += agent.output_tokens_used
            except Exception as exc:
                print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            cost = tokens[0] * in_p / 1e6 + tokens[1] * out_p / 1e6
            if cost > args.max_cost:
                print("[abort] max-cost reached", flush=True)
                break

    hall_flagged = [t for t in results if labels_before[t] == 1]
    clean_flagged = [t for t in results if labels_before[t] == 0]
    mitigated = sum(1 for t in hall_flagged if results[t] == 0)
    harmed = sum(1 for t in clean_flagged if results[t] == 1)
    cost = tokens[0] * in_p / 1e6 + tokens[1] * out_p / 1e6

    lines = [
        f"# e5_intervention  model={args.model} seed={args.seed} "
        f"flag_frac={args.flag_frac} n_test={len(test_ids)}",
        f"flagged: {len(results)} rerun ({len(hall_flagged)} hallucinating, "
        f"{len(clean_flagged)} clean at flag time)",
        f"mitigation: {mitigated}/{len(hall_flagged)} flagged-hallucinating sessions "
        f"became clean ({mitigated / max(len(hall_flagged), 1):.1%})",
        f"collateral: {harmed}/{len(clean_flagged)} flagged-clean sessions became "
        f"hallucinating ({harmed / max(len(clean_flagged), 1):.1%})",
        f"net label change on test split: "
        f"{(mitigated - harmed)}/{len(test_ids)} sessions "
        f"({(mitigated - harmed) / max(len(test_ids), 1):.1%} absolute reduction)",
        f"cost: ${cost:.3f}",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    np.savez(
        OUT_DIR / f"{args.model}_{args.seed}.npz",
        flagged=np.array(list(results)),
        label_before=np.array([labels_before[t] for t in results]),
        label_after=np.array([results[t] for t in results]),
        cost=cost,
    )
    print("\n".join(lines))
    print(f"({time.time()-t0:.0f}s) [ledger] record in results/RESULTS.md")


if __name__ == "__main__":
    main()
