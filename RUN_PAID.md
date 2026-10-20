# RUN_PAID.md: paid LLM runs (confirmatory plan)

One launcher runs everything in priority order with hard caps. The analysis
plan is fixed in `results/PREREG_confirmatory.md`, committed before any of
this data existed.

```
DRY=1 bash run_paid.sh     # $0: prints every command and its cost estimate
bash run_paid.sh           # the real run (~3-4 h unattended, estimated ~$37 if every step runs)
```

Keys: in `.env` at the repo root (read by `pge/config.py`), or exported:

```
TOGETHER_API_KEY=...     # Llama-3.3-70B agent
OPENROUTER_API_KEY=...   # Llama-3.1-8B (agent, simulated patient, reply labeler), Qwen2.5-72B, GPT-4o-mini, Mistral-Small-3.1-24B
```

## Steps (priority order)

| id | what | estimate | cap |
|---|---|---|---|
| 1a | confirmatory corpus, 70B, n=600, seed 7, protocol v2fb, k=3 resamples on every turn | $9.93 | $12.50 |
| 1b | same, Llama-3.1-8B | $0.56 | $1.50 |
| 1c | same, Qwen2.5-72B | $4.51 | $6.00 |
| 1d | same, GPT-4o-mini | $1.89 | $2.50 |
| 1e | same, Mistral-Small-3.1-24B | $4.06 | $5.50 |
| 2a | LLM-simulated patient pilot, patient knows only its own history, 100 sessions | $0.66 | $1.20 |
| 2b | same, patient also told the true decision ("informed"), 100 sessions | $0.66 | $1.20 |
| 2c | profile arm extended to 600 (reuses the 100 pilot sessions), only if the 2a audit lift >= 1.5 | ~$3.40 | $5.00 |
| 3 | measured closed-loop clarification on the 1a corpus (180 test sessions, 9 conditions) | ~$6.60 | $9.00 |
| 4 | CoVe, 70B self-verification, on the 1a corpus | ~$0.87 | $1.50 |
| 5a | OPTIONAL: informed arm extended to 600, if its pilot passes | ~$3.40 | $5.00 |
| 5b | OPTIONAL: k=10 resamples, user-dependent turns, 1a split-seed-42 test sessions | ~$2.90 | $4.00 |

`TOTAL_CAP` (default $41 = $50 budget - $7.96 already spent - ~$1 margin)
bounds recorded spend across all runs of the launcher on this machine
(`logs/paid/spend.tsv`): a step is refused if recorded spend + its cap would
exceed it, so if money runs short only the optional tail is dropped. Expected
total if everything runs: ~$37 here, ~$45 including earlier spend.

## What changed from the earlier plan (and why)

The 2026-10-20 blind review found that the v2 agent was shown a corrupted
feedback display (no reaction as "+", inverted below p = 0.5), that resamples
existed only for 5 attributes, and that the composite label is mostly not
user-specific. So every corpus is now generated fresh under protocol `v2fb`
(raw reaction shown, resamples on every turn) on a seed never used in
development, and analysed under the pre-registered plan with a user-specific
label. The development corpora (seed 42) remain the development evidence.

## Semantics to know (state in the paper where noted)

- **`--max-cost` is per invocation**; the launcher's TOTAL_CAP is the global guard.
- **Stops** on any error and on any `[abort]` (cost cap, parse-failure guard
  >25% after 10 sessions, repeated HTTP 4xx). Partial output is kept as
  `<output>.partial.jsonl`; rerunning `bash run_paid.sh` skips finished steps
  and resumes the stopped one.
- **Protocols (paper must say).** `v2` = development corpora, agent saw
  sign(tanh(F logit p)). `v2fb` = confirmatory, agent sees "+"/"-"/"none".
  Every script that rebuilds an agent prompt from a corpus (clarification,
  CoVe, resample replay) reads `meta.protocol` and uses the same display.
- **Parsing (paper must say).** The routed agent parses leniently and
  re-prompts once when the value is outside the allowed set (the v2 agent
  accepted such values). Per-turn status in `meta.parse_status`; a failed
  turn keeps `unknown`, p = 0.5, flagged in `meta.parse_failed_turns`.
- **Labels.** `eval_label` = composite label; `meta.eval_label_user` = the
  user-specific label L_U (truth-derived, never read by detectors).
- **LLM patient.** In the default arm the patient knows its own history only,
  never the clinic's decision. The reply labeler sees only the statement and
  the reply. Replies and labeler outputs are in `meta.turn_audit`; the
  patient prompt (which holds truth-derived text) is never stored.
- **Clarification.** V6 (pre-registered detector) ranks commitments; answers
  enter the agent prompt from each picked turn onward; controls: no_question@b,
  rerun_full, random@1, lowconf@1; outcomes on both labels.

## After the runs: send back

Commit and push (or zip and attach):
- `logs/paid/`
- `data/trajectories/healthcare_confirm/` (`*.jsonl`, `*.run.json`)
- `data/trajectories/healthcare_llmpatient/` (`pilot/`, `70B_7*.jsonl`, `*.run.json`)
- `data/trajectories/healthcare_confirm_k10/` if step 5b ran
- `results/e8_closed_loop/70B_7/`, `results/cove_baseline/70B_on_70B_7/`
- any `*.partial.jsonl` left behind

## Risks to check in the logs

- **Model ids** fail at the 3-token preflight call (~$0) if a provider stops serving them.
- **Prices** of the new families are list prices marked "verify on provider site"
  in `pge/agents/registry.py`; the caps hold regardless.
- **8B patient/labeler quality**: watch "patient replies with verdict words" and
  the audit's P(-1|w) vs P(-1|r).
- **Rate limits**: HTTP 429 is retried with backoff; lower `--concurrency` if Together throttles step 1a.

## Offline dry run (what was tested)

```
# whole launcher, in a scratch COPY of the repo only (it writes data/ and results/):
PAID_EXTRA_ARGS="--mock" PGE_MOCK_ANY_OUT=1 bash run_paid.sh
```
