# RUN_PAID.md: paid LLM runs

One launcher runs everything in the recommended order with hard caps:

```
DRY=1 bash run_paid.sh     # $0: prints every command and its cost estimate
bash run_paid.sh           # the real run (~2.4 h, estimated ~$26, sum of caps $36.90 <= $40)
```

Every paid script (`eval/paid/*`):
- takes a hard `--max-cost`. Before each API call the worst-case cost of that call
  (estimated prompt tokens + `max_tokens` at the model's price) is reserved, and the call is
  refused if spent + in-flight + this call would exceed the cap. Actual cost is settled from
  the provider's usage block, or the provider-reported cost if that is larger. A timed-out
  call is charged its full reservation (the provider may have billed it).
- records tokens and cost (per role: agent / patient / extractor / verifier), writes
  trajectories through `pge.trajectory.save_jsonl` (non-circularity lint), writes a
  `<output>.run.json` with args, model, prices, meter and parse statistics, and prints a
  `[ledger] | date | ... |` row to paste into `results/RESULTS.md`.
- makes one 3-token preflight call per endpoint first, so a missing key or a model id the
  provider does not serve fails immediately at ~$0.
- supports `--estimate-only` ($0), `--resume` (continue from `<output>.partial.jsonl`), and
  `--mock` (offline dry run, requires `--out`).

## Keys

`.env` at the repo root (read by `pge/config.py`), or exported in the shell:

```
TOGETHER_API_KEY=...     # Together: Llama-3.3-70B agent (steps 1, 3, 4, 5)
OPENROUTER_API_KEY=...   # OpenRouter: Llama-3.1-8B patient/extractor (step 1), Qwen2.5-72B, Mistral-Small-3.1-24B, GPT-4o-mini (step 2)
```

## Steps (run in this order by `run_paid.sh`)

Prices: the 70B and 8B rates are the ledger's. The new families' prices are list prices
marked "verify on provider site" in `pge/agents/registry.py`, so check them before running.
Wall times are extrapolated from the ledger runs: the e5 reruns took 46 s for 530 70B calls
with 8 workers, and the 8B augmentation took 246 s for 2,478 calls.

| id | what | estimate | cap | wall | output |
|---|---|---|---|---|---|
| 1a | LLM-patient pilot, patient knows only its own profile (default arm), 100 sessions | $0.68 | $1.20 | ~5 min | `data/trajectories/healthcare_llmpatient/pilot/70B_42_n100.jsonl` |
| 1b | LLM-patient pilot, `informed` arm (patient also told the true decision), 100 sessions | $0.68 | $1.20 | ~5 min | `.../pilot/70B_42_n100_informed.jsonl` |
| 1c | n=600 profile arm, **only if** the 1a audit gate passes | ~$3.40 (500 new) | $5.00 | ~25 min | `data/trajectories/healthcare_llmpatient/70B_42.jsonl` |
| 1d | n=600 informed arm, **only if** the 1b audit gate passes | ~$3.40 | $5.00 | ~25 min | `.../70B_42_informed.jsonl` |
| 2a | healthcare-v2, Qwen2.5-72B (OpenRouter), n=600 | $1.30 | $2.50 | ~15 min | `data/trajectories/healthcare_v2_models/qwen72b_42.jsonl` |
| 2b | healthcare-v2, Mistral-Small-3.1-24B (OpenRouter), n=600 | $1.19 | $2.00 | ~15 min | `.../mistral-small-24b_42.jsonl` |
| 2c | healthcare-v2, GPT-4o-mini (OpenRouter), n=600 | $0.58 | $1.50 | ~12 min | `.../gpt-4o-mini_42.jsonl` |
| 3 | measured closed-loop clarification, 180 test sessions x {detailed_questions, rerun_full, posterior@1, no_question@1, posterior@2, no_question@2, random@1, lowconf@1} | $7.16 | $9.00 | ~15 min | `results/e8_closed_loop/70B_42/` |
| 4 | k=10 resamples on healthcare 70B (2,483 turns, +17,381 samples) | $7.04 | $8.00 | ~20 min | `data/trajectories/healthcare_v2_k10/70B_42_k10.jsonl` |
| 5 | CoVe baseline, 70B self-verification, test-union of split seeds 42/123/456 (394 tasks, 3,892 queries) | $0.86 | $1.50 | ~6 min | `results/cove_baseline/70B_on_70B_42/` |
| | **total** | **~$26** | **$36.90** | **~2.4 h** | |

Ledger: cumulative spend is $7.96 now, so this batch ends at ~$34. Even if every step hit its
cap, the total would be $44.86, still under the $45 stop-gate in `results/RESULTS.md`.

Cheaper variants:
- Step 4 with `--restrict test-union` costs $4.61 instead of $7.04. It augments only the 394
  test-union tasks, so train splits mix k=3 and k=10. That is fine for zero-shot SelfCheckGPT
  and semantic entropy, but not for the V4 posterior's sample sensor.
- Step 5 with `--model gpt-4o-mini` costs $0.17 (not self-verification).
- Step 2 with `--model qwen72b-together` (Together Qwen Turbo) costs $3.90, versus $1.30 for the
  same weights on OpenRouter.

Exact single-step commands (what `run_paid.sh` runs; every one also takes `--resume`):

```
python3 -m eval.paid.generate_healthcare_llmpatient --model 70B --seed 42 --n 100 --patient-knows profile  --out data/trajectories/healthcare_llmpatient/pilot/70B_42_n100.jsonl --max-cost 1.2
python3 -m eval.paid.generate_healthcare_llmpatient --model 70B --seed 42 --n 100 --patient-knows informed --out data/trajectories/healthcare_llmpatient/pilot/70B_42_n100_informed.jsonl --max-cost 1.2
python3 -m eval.paid.audit_llmpatient data/trajectories/healthcare_llmpatient/pilot/70B_42_n100.jsonl --seed 42 --gate-lift 1.5   # exit 0 = pass, 3 = fail
python3 -m eval.paid.generate_healthcare_llmpatient --model 70B --seed 42 --n 600 --patient-knows profile  --out data/trajectories/healthcare_llmpatient/70B_42.jsonl --max-cost 5 --resume
python3 -m eval.paid.generate_healthcare_llmpatient --model 70B --seed 42 --n 600 --patient-knows informed --out data/trajectories/healthcare_llmpatient/70B_42_informed.jsonl --max-cost 5 --resume
python3 -m eval.paid.generate_healthcare_multimodel --model qwen72b           --seed 42 --n 600 --max-cost 2.5
python3 -m eval.paid.generate_healthcare_multimodel --model mistral-small-24b --seed 42 --n 600 --max-cost 2.0
python3 -m eval.paid.generate_healthcare_multimodel --model gpt-4o-mini       --seed 42 --n 600 --max-cost 1.5
python3 -m eval.paid.run_clarification_closed_loop --model 70B --seed 42 --max-cost 9
python3 -m eval.paid.augment_resamples_k10 --model 70B --seed 42 --k-target 10 --max-cost 8
python3 -m eval.paid.run_cove --model 70B --max-cost 1.5
```

An unregistered model needs an explicit provider and prices, for example
`--model some/model --provider openrouter --price-in 0.2 --price-out 0.2 --short-name mymodel`.

## Semantics to know (and to state in the paper where noted)

- **`--max-cost` is per invocation.** A `--resume` run starts its meter at $0. `run_paid.sh`
  adds a global guard on top: it appends each step's actual spend to `logs/paid/spend.tsv`
  (or the full cap if a step crashed before reporting) and refuses to start a step when
  recorded spend + that step's cap would exceed `TOTAL_CAP` (default 40). That check holds
  across reruns of the launcher. Delete `spend.tsv` only if you mean to reset it.
- **Stop behavior.** The launcher stops on any error and on any `[abort]`: a cost-cap stop,
  the parse-failure guard (`--max-parse-fail`, default 25% of turns after 10 sessions), or
  repeated HTTP 4xx. Partial output is kept as `<output>.partial.jsonl`. After a fix, rerun
  `bash run_paid.sh`: finished steps are skipped and the stopped one resumes. If a step hit
  its cap, raise the cap only after checking the cost per session in its log.
- **Parsing differs from the 2026-09 corpora (paper setup must say so).** The routed agent sends
  byte-identical prompts, but it (a) parses leniently (markdown, missing brackets, `p=70%`,
  `probability: 0.7`, values with spaces), and (b) **re-prompts with the same format reminder
  when the value is outside the allowed set**. The legacy agent accepted out-of-set values and
  scored them as wrong. Per-turn status is in `meta.parse_status`:
  `strict | strict_normalized | lenient | lenient_no_attr | retry1_* | strict_invalid_value | failed`.
  A `failed` turn keeps the legacy placeholder (`unknown`, p=0.5, below the label threshold),
  is listed in `meta.parse_failed_turns`, and its feedback is the environment's honest
  reaction to an unusable commitment. A `strict_invalid_value` turn keeps the model's own
  out-of-set value, which is wrong by construction. Neither is ever mapped onto a valid value.
- **LLM patient.**
  - The patient model gets its own history (setup + hidden fields) and, in the default
    `profile` arm, never the clinic's decision: it can catch an error only by connecting it to
    what it knows about itself. In the `informed` arm it is also told the true decision.
  - The extractor sees only the agent's statement and the patient's reply, and outputs
    AGREES / CONTRADICTS / NONE, stored as the `patient_feedback` signal +1 / -1 / 0. An
    unparseable extractor output gets one retry; if that also fails, the signal is 0 and the
    turn is listed in `meta.extractor_failed_turns`.
  - NO_FEEDBACK_ATTRS get no patient call, as in v2. `--feedback-on-all` changes that.
  - Patient replies and raw extractor outputs are in `meta.turn_audit` (not detector-visible).
  - The patient prompt itself holds the truth and is never stored.
  - The n=600 run reuses the 100 pilot sessions. Same config and session ids: they are seeded
    into `<output>.partial.jsonl` and resumed. That run's ledger row covers only the 500 new
    sessions, so add the pilot row.
- **Clarification (step 3).**
  - Picks for posterior@b are the b confident commitments with the highest q_t under
    `IndependentPosterior(by_value=True)` fit on the seed-42 train split. These are identical
    to the offline `eval/e8_clarification.py` posterior@b picks.
  - The patient's answers enter the agent context as first-person statements, from the turn of
    each pick onward. detailed_questions puts them in from turn 0.
  - Controls: no_question@b reruns from the posterior@b position with no answer;
    rerun_full reruns the whole session with no answer (control for detailed_questions);
    random@1 and lowconf@1 ask one commitment chosen at random or by lowest stated
    confidence (does the posterior's choice matter?).
  - Every rerun of session i uses the e5 per-session noise stream `default_rng((42, 7919, i))`.
  - Done marker for the launcher: the per-condition JSONL files, written only when every
    condition is complete (summary.txt is rewritten on partial runs too).
  - Answer text (truth) is never stored; `meta.questions` lists attributes and turns only.
  - Outcome: `eval_label_v2` on the resulting commitments, plus wrong confident commitments
    per session, with paired bootstrap CIs in `summary.txt`.
- **CoVe (step 5).** The healthcare caches store no per-turn prompt. The runner therefore
  rebuilds each turn's context from detector-visible inputs only: intake answers, attribute
  definition, and prior commitments with their feedback marks. It runs `cove.py` unchanged at
  T=0. Responses are cached in `cove_cache.jsonl`, keyed by (task_id, sha256(model + prompt)),
  so a rerun is free. `cove.py` counts an unparseable re-answer as a disagreement; the count is
  printed and saved as `unparsed_replies`. `scores.json` is written only when every split is
  complete (otherwise `scores.partial.json`), so a capped run resumes on rerun.

## After the runs ($0, offline)

```
# LLM-patient: channel audit (truth reconstructed from the seed stream; analysis only)
python3 -m eval.paid.audit_llmpatient data/trajectories/healthcare_llmpatient/70B_42.jsonl --seed 42 --out results/llmpatient/audit_70B_42.txt
python3 -m eval.paid.audit_llmpatient data/trajectories/healthcare_llmpatient/70B_42_informed.jsonl --seed 42 --out results/llmpatient/audit_70B_42_informed.txt
# e6 detectors + ChannelPosterior (V6) on every new corpus (same splits/seeds/bootstrap as e6)
python3 -m eval.paid.score_new_corpus data/trajectories/healthcare_llmpatient/70B_42.jsonl          > results/llmpatient/scores_70B_42.txt
python3 -m eval.paid.score_new_corpus data/trajectories/healthcare_llmpatient/70B_42_informed.jsonl > results/llmpatient/scores_70B_42_informed.txt
python3 -m eval.paid.score_new_corpus data/trajectories/healthcare_v2_models/qwen72b_42.jsonl           > results/e10_multimodel/qwen72b.txt
python3 -m eval.paid.score_new_corpus data/trajectories/healthcare_v2_models/mistral-small-24b_42.jsonl > results/e10_multimodel/mistral-small-24b.txt
python3 -m eval.paid.score_new_corpus data/trajectories/healthcare_v2_models/gpt-4o-mini_42.jsonl       > results/e10_multimodel/gpt-4o-mini.txt
# k=10 sampling baselines
python3 -m eval.e6_posterior healthcare data/trajectories/healthcare_v2_k10/70B_42_k10.jsonl > results/e6_posterior/healthcare_70B_k10.txt
```

(The `results/` subdirectories above are suggestions; `mkdir -p` them first. Each scoring run takes about 1 min.)
CoVe scores: `results/cove_baseline/70B_on_70B_42/scores.json` has task_id -> score for every
test-union task, and per split seed the test task_ids, scores and AUROC. `seed<k>.npz` holds
`task_ids`, `score_cove` and `labels`, ready to merge into the e6 tables.

## What to send back

- `logs/paid/` (every step log, `spend.tsv`, pilot audits `audit_*.txt`)
- `data/trajectories/healthcare_llmpatient/` (pilot/, `70B_42*.jsonl`, `*.run.json`)
- `data/trajectories/healthcare_v2_models/*.jsonl` and `*.run.json`
- `data/trajectories/healthcare_v2_k10/70B_42_k10.jsonl` and `.run.json`
- `results/e8_closed_loop/70B_42/` (summary.txt, sessions.json, per-condition JSONL)
- `results/cove_baseline/70B_on_70B_42/` (scores.json, seed*.npz, cove_cache.jsonl, cove_inputs.jsonl)
- any `*.partial.jsonl` left behind, plus the `[ledger]` lines from the logs

## Risks to check in the logs

- **Model ids.** `qwen/qwen-2.5-72b-instruct`, `mistralai/mistral-small-3.1-24b-instruct`
  and `openai/gpt-4o-mini` are current OpenRouter slugs. Together may have retired
  `Qwen/Qwen2.5-72B-Instruct-Turbo` from serverless, as it did with 8B; it is not in the
  default plan. A wrong id fails at preflight (HTTP 400/404) for ~$0.
- **8B patient quality.** Watch `patient replies with verdict words` and the audit's
  P(-1|w) vs P(-1|r). If the 8B extractor is unreliable (high `failed` / `lenient` counts),
  rerun the extractor role with `--extractor-model gpt-4o-mini` (+~$0.15 per 600 sessions).
- **Rate limits.** HTTP 429 is retried with backoff (4 s x 1.6^k, 4 retries). If Together
  throttles step 4, lower `--concurrency` (default 12).

## Offline dry run (what was tested)

```
D=/path/to/scratch
python3 -m eval.paid.generate_healthcare_multimodel --model qwen72b --n 60 --mock --out $D/mm/qwen72b_42.jsonl
python3 -m eval.paid.generate_healthcare_llmpatient --n 60 --mock --out $D/lp/70B_42.jsonl
python3 -m eval.paid.augment_resamples_k10 --mock --out $D/k10/70B_42_k10.jsonl
python3 -m eval.paid.run_clarification_closed_loop --mock --out $D/clar
python3 -m eval.paid.run_cove --mock --out $D/cove
# whole launcher, in a scratch COPY of the repo only (it writes data/ and results/):
PAID_EXTRA_ARGS="--mock" PGE_MOCK_ANY_OUT=1 bash run_paid.sh
```
