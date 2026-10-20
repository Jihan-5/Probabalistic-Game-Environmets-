"""Healthcare with an LLM-simulated patient: behavioral feedback (PAID).

See pge/benchmarks/healthcare_llmpatient.py for the protocol. Per turn:
agent commits -> commitment read to the PATIENT LLM (knows its own history;
with --patient-knows informed also the true value) -> 1-2 sentence natural reply -> EXTRACTOR LLM (sees only
statement + reply) -> signal `patient_feedback` in {+1, -1, 0}. The agent's
next prompt shows the patient's replies (not +/-). eval_label = eval_label_v2
on noise-free truth. Same session stream / task ids as healthcare-v2.

By default attributes in NO_FEEDBACK_ATTRS get no patient call (signal 0,
"(no reply)"), mirroring v2; --feedback-on-all lets the patient react to
every commitment.

Usage:
    python3 -m eval.paid.generate_healthcare_llmpatient --model 70B --seed 42 --n 600 --max-cost 6
Writes data/trajectories/healthcare_llmpatient/<agent short>_<seed>.jsonl (profile arm) or
<agent short>_<seed>_informed.jsonl (informed arm), each with .run.json.
Audit: python3 -m eval.paid.audit_llmpatient <jsonl> --seed 42
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from eval.paid.common import (
    ALLOWED_VALUES, REPO, add_dag_node, add_model_args, add_run_args, check_mock_out,
    finalize, is_failed, ledger_line, load_partial, make_client, preflight, print_estimate,
    rel, run_parallel, save_partial, spec_from_args, write_run_json,
)
from pge.agents.paid_client import ChatClient, CostMeter, RoutedAgent
from pge.benchmarks import healthcare_llmpatient as LP
from pge.benchmarks.healthcare_v2 import (
    ATTR_DESCRIPTIONS_V2, NO_FEEDBACK_ATTRS, AgentGuessV2, SessionV2, eval_label_user,
    eval_label_v2, sample_session_v2,
)
from pge.dag import ChainOfThoughtDAG
from pge.trajectory import Trajectory, Turn

OUT_DIR = REPO / "data" / "trajectories" / "healthcare_llmpatient"
SYSTEM_PROMPT_LP = (REPO / "pge" / "prompts" / "healthcare_llmpatient_system.txt").read_text()

PATIENT_TEMP, PATIENT_MAX_TOKENS = 0.7, 80
EXTRACTOR_TEMP, EXTRACTOR_MAX_TOKENS = 0.0, 8
# token estimates per call (agent prompt grows with the replies in history)
EST = {"agent": (650, 30, 1.1), "patient": (480, 40, 1.0), "extractor": (190, 3, 1.05)}


class PatientDialogueAgent(RoutedAgent):
    """RoutedAgent whose history shows the patient's verbatim replies instead
    of +/- feedback marks. Prompt template otherwise identical to v2."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.replies: dict[int, str | None] = {}  # turn -> patient reply (None = no reply)

    def _format_dialogue(self, history: ChainOfThoughtDAG, max_turns: int = 6) -> str:
        if len(history) == 0:
            return "(no prior guesses)"
        lines = []
        for node in history.nodes[-max_turns:]:
            reply = self.replies.get(node.turn)
            lines.append(f"  turn {node.turn}: [{node.attribute}]: {node.value}, p={node.probability:.2f}")
            lines.append(f'    patient replied: "{reply}"' if reply else "    (no reply from the patient)")
        return "\n".join(lines)

    def _build_user_prompt(self, attribute, setup_answers, history):
        setup_str = "\n".join(f"  {k}: {v}" for k, v in setup_answers.items())
        description = self.attribute_descriptions.get(attribute, "")
        desc_line = f"\nThis attribute means: {description}\n" if description else ""
        return (
            f"Context (one-time, from the user/task):\n{setup_str}\n\n"
            f"Prior turns in this session:\n{self._format_dialogue(history)}\n\n"
            f"Now produce your guess for attribute: {attribute}{desc_line}\n"
            f"Output format: [{attribute}]: <value>, p=<probability>"
        )


def extract(extractor: ChatClient, statement: str, reply: str) -> tuple[int, str, list[str]]:
    """(signal, status, raw outputs). One retry on an unparseable label; if
    that fails too: signal 0 (unobserved), status 'failed'."""
    raws = []
    for retry in (False, True):
        raw = extractor.chat(LP.extractor_messages(statement, reply, retry=retry),
                             EXTRACTOR_TEMP, EXTRACTOR_MAX_TOKENS)
        raws.append(raw)
        sig, status = LP.parse_extractor(raw)
        if sig is not None:
            return sig, (status if not retry else f"retry1_{status}"), raws
    return 0, "failed", raws


def run_patient_session(session: SessionV2, agent: PatientDialogueAgent, patient: ChatClient,
                        extractor: ChatClient, model_id: str, seed: int, persona: str,
                        feedback_on_all: bool, knows: str = "profile") -> Trajectory:
    G = ChainOfThoughtDAG()
    node_idx: dict[str, int] = {}
    turns: list[Turn] = []
    guesses: dict[str, AgentGuessV2] = {}
    audit = []

    for t, attribute in enumerate(session.schedule):
        guess, raw = agent.guess_full(t, session.setup_answers, G)
        status = agent.last_parse_status
        value = str(guess.value).strip().lower()
        guesses[attribute] = AgentGuessV2(attribute, value, guess.probability)
        statement = LP.commitment_sentence(attribute, value, guess.probability)

        entry = {"turn": t, "attribute": attribute, "statement": statement, "persona": persona}
        if attribute in NO_FEEDBACK_ATTRS and not feedback_on_all:
            signal, reply = 0, None
            entry.update(patient_reply=None, extractor_raw=[], extractor_status="no_feedback_attr")
        else:
            reply = patient.chat(LP.patient_messages(session, attribute, statement, persona, knows),
                                 PATIENT_TEMP, PATIENT_MAX_TOKENS).strip()
            signal, ex_status, ex_raw = extract(extractor, statement, reply)
            entry.update(patient_reply=reply, extractor_raw=ex_raw, extractor_status=ex_status,
                         patient_verdict_word=bool(LP.VERDICT_RE.search(reply)))
        agent.replies[t] = reply
        entry["agent_parse_status"] = status
        audit.append(entry)

        turns.append(Turn(attribute=attribute, guess_value=value, probability=guess.probability,
                          turn=t, raw_response=raw, signals={LP.SIGNAL: signal}))
        add_dag_node(G, node_idx, attribute, value, guess.probability, signal, t)

    return Trajectory(
        task_id=f"hcv2_{session.profile_id}",
        domain=LP.DOMAIN,
        model=model_id,
        seed=seed,
        turns=turns,
        eval_label=eval_label_v2(guesses, session),
        label_source="programmatic",
        meta={
            "channel": "llm_patient",
            "eval_label_user": eval_label_user(guesses, session),  # truth-derived label
            "n_turns": len(session.schedule),
            "persona": persona,
            "patient_knowledge": knows,
            "feedback_on_all": feedback_on_all,
            "parse_status": [a["agent_parse_status"] for a in audit],
            "parse_failed_turns": [a["turn"] for a in audit if is_failed(a["agent_parse_status"])],
            "extractor_failed_turns": [a["turn"] for a in audit if a["extractor_status"] == "failed"],
            "turn_audit": audit,  # raw patient replies + extractor outputs (NOT detector-visible)
        },
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, default="70B", help_role="agent")
    add_model_args(ap, prefix="patient-", default="8B", help_role="patient")
    add_model_args(ap, prefix="extractor-", default="8B", help_role="extractor")
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feedback-on-all", action="store_true",
                    help="patient also reacts to NO_FEEDBACK_ATTRS (default: mirror v2)")
    ap.add_argument("--max-parse-fail", type=float, default=0.25)
    ap.add_argument("--patient-knows", choices=LP.PATIENT_KNOWLEDGE, default="profile",
                    help="profile: patient knows only its own history (default); "
                         "informed: also told the true decision (comparison arm)")
    add_run_args(ap, max_cost=6.0)
    args = ap.parse_args(argv)
    check_mock_out(args)

    a_spec = spec_from_args(args, short_name=args.short_name)
    p_spec = spec_from_args(args, prefix="patient_")
    e_spec = spec_from_args(args, prefix="extractor_")
    suffix = "" if args.patient_knows == "profile" else f"_{args.patient_knows}"
    out = Path(args.out) if args.out else OUT_DIR / f"{a_spec.short_name}_{args.seed}{suffix}.jsonl"

    rng = np.random.default_rng(args.seed)
    sessions = [sample_session_v2(rng, i) for i in range(args.n)]
    personas = [LP.sample_persona(args.seed, i) for i in range(args.n)]
    n_turns = sum(len(s.schedule) for s in sessions)
    n_fb = sum(1 for s in sessions for a in s.schedule
               if args.feedback_on_all or a not in NO_FEEDBACK_ATTRS)
    est = print_estimate(
        f"healthcare LLM-patient n={args.n} ({n_turns} turns, {n_fb} patient replies) -> {rel(out)}",
        [("agent turns", a_spec, int(n_turns * EST["agent"][2]), *EST["agent"][:2]),
         ("patient replies", p_spec, int(n_fb * EST["patient"][2]), *EST["patient"][:2]),
         ("extractor labels", e_spec, int(n_fb * EST["extractor"][2]), *EST["extractor"][:2])],
        args.max_cost,
    )
    if args.estimate_only:
        return

    meter = CostMeter(args.max_cost)
    agent_c = make_client(a_spec, meter, "agent", args)
    patient_c = make_client(p_spec, meter, "patient", args)
    extractor_c = make_client(e_spec, meter, "extractor", args)
    preflight([agent_c, patient_c, extractor_c])

    done = {t.task_id: t for t in load_partial(out)} if args.resume else {}
    if done:
        print(f"[resume] {len(done)} sessions already in {rel(out)}.partial.jsonl", flush=True)
    todo = [i for i in range(args.n) if f"hcv2_{i}" not in done]
    statuses: Counter = Counter(s for t in done.values() for s in t.meta.get("parse_status", []))

    def one(i: int):
        agent = PatientDialogueAgent(
            attribute_schedule=sessions[i].schedule, client=agent_c, allowed_values=ALLOWED_VALUES,
            attribute_descriptions=ATTR_DESCRIPTIONS_V2, temperature=0.3, max_tokens=200,
            system_prompt=SYSTEM_PROMPT_LP, request_timeout=45.0,
        )
        return run_patient_session(sessions[i], agent, patient_c, extractor_c, a_spec.model_id,
                                   args.seed, personas[i], args.feedback_on_all, args.patient_knows)

    def on_result(i, traj):
        done[traj.task_id] = traj
        statuses.update(traj.meta["parse_status"])
        total = sum(statuses.values())
        if len(done) >= 10 and total and statuses["failed"] / total > args.max_parse_fail:
            if not meter.tripped:
                print(f"[abort] agent parse-failure rate {statuses['failed']/total:.1%}", flush=True)
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
              "rerun with --resume (the new run's --max-cost starts from $0)", flush=True)

    labels = np.array([t.eval_label for t in results])
    sig = Counter(tt.signals[LP.SIGNAL] for t in results for tt in t.turns)
    ex = Counter(a["extractor_status"] for t in results for a in t.meta["turn_audit"])
    verdict = sum(1 for t in results for a in t.meta["turn_audit"] if a.get("patient_verdict_word"))
    total = sum(statuses.values())
    summ = meter.summary()
    print(f"\n[done] {n_written} trajectories -> {rel(out)}{'' if complete else '.partial.jsonl'}"
          f"  ({stats['seconds']:.0f}s)")
    print(f"[labels] hallucinating {labels.sum()}/{len(labels)} = {labels.mean():.1%}")
    print(f"[signal] patient_feedback counts {{+1: {sig[1]}, -1: {sig[-1]}, 0: {sig[0]}}}")
    print(f"[parse] agent {dict(statuses)} (failed {statuses['failed']}/{total}); "
          f"extractor {dict(ex)}; patient replies with verdict words: {verdict}")
    print(f"[cost] ${summ['cost']:.3f} in={summ['tokens_in']} out={summ['tokens_out']} "
          f"calls={summ['calls']} (estimate ${est:.2f}); by role: "
          + ", ".join(f"{k} ${v['cost']:.3f}" for k, v in summ["by_role"].items()))
    write_run_json(out, {"script": "generate_healthcare_llmpatient", "args": vars(args),
                         "agent": a_spec.__dict__, "patient": p_spec.__dict__,
                         "extractor": e_spec.__dict__, "complete": complete, "n_written": n_written,
                         "parse_status_counts": dict(statuses), "extractor_status_counts": dict(ex),
                         "signal_counts": {str(k): v for k, v in sig.items()},
                         "hall_rate": float(labels.mean()), "meter": summ, "runner": stats})
    print("[ledger] " + ledger_line(
        f"healthcare LLM-patient corpus ({a_spec.short_name}, seed {args.seed}, "
        f"patient knows {args.patient_knows}, n={n_written})",
        f"behavioral feedback: patient {p_spec.model_id}, extractor {e_spec.model_id}; "
        f"signal +1/-1/0 = {sig[1]}/{sig[-1]}/{sig[0]}; agent parse-failed {statuses['failed']}/{total}, "
        f"extractor failed {ex['failed']}",
        f"{n_written} trajectories, {labels.mean():.1%} hallucinating", summ["cost"], rel(out) + ("" if complete else ".partial.jsonl")))


if __name__ == "__main__":
    main()
