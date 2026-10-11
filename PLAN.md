# PGE Resubmission Rebuild Plan (AISTATS/ICML 2027)

> **REVISION LOG (2026-09-13)** — changes since approval, in execution order:
> 1. **Target locked: AISTATS 2027** (abstract Sept 29, full paper Oct 6, 2026 — user chose the sprint). τ-bench third domain demoted to stretch goal; volume comes from healthcare-v2 (LLM) + all ~118 usable dbbench tasks × models × seeds with task-level splits.
> 2. **Gate G1 = NO-GO** (graphon 0.61–0.67 vs dag_logreg 0.78–0.82 on its own best-case synthetic env) → **framing B adopted**: protocol + corrected theory + systematic detector study; graphon reported as honest negative. Confirmed on real data (both domains: graphon ≤ chance).
> 3. **Theory correction extended**: the rebuttal's own claim "p̂* < q for all q" is false — conservative only for q < 1 − α/(βγ) (= 5/6 at defaults), saturating at q ≥ βγ/(α+βγ). Exact lemma implemented + tested in `pge/scoring.py`.
> 4. **CoT→DAG LLM extraction de-scoped** for the sprint: DAGs come from the protocol's known attribute-dependency structure (deterministic, more defensible than LLM parsing); reasoning text still captured for the worked example.
> 5. **dbbench corpus is hard-capped at ~120 public SELECT-style tasks** → widened to 8 task types ≤12 cols, type-aware answer probes; accuracy fixed 13.9% → 39%.
> 6. **Multi-agent**: plain graphon-MLCR proven ≡ majority vote (op-independent likelihood — documented degeneracy). Replaced by **ReliabilityMLCR** (confidence × cascade blocks, learned reliabilities, weights floored at 0): 0.685 vs 0.662 majority, McNemar p=0.047 on 784 conflicts. Vote-inverting variant measured and rejected (0.592).
> 7. **Model swap**: Together retired serverless Llama-8B → 8B tier now `meta-llama/llama-3.1-8b-instruct` via OpenRouter; 70B unchanged (Together Llama-3.3-70B-Turbo).
> 8. **Healthcare resample augmentation** added (replay cached contexts, k=3 samples on underdetermined attributes) so sampling baselines are fair on all domains.
> 9. Execution is multi-agent + autonomous (user request): live narrative in repo `STATUS.md`, numbers-of-record in repo `results/RESULTS.md`.

## Context

The ICML 2026 PGE submission was rejected. The AC cited: an erroneous incentive-compatibility proof never formally corrected, an unsupported 96.4% multi-agent claim, pervasive presentation issues, reproducibility gaps, and a conflated graphon ablation. This repo is the **only** codebase, and its own results contradict the paper: the graphon log-likelihood detector scores **0.20–0.55 AUROC (at/below chance)** on every real run, while the ~1.0 AUROC numbers come only from circular detectors whose input signal is computed from the same ground truth used as the eval label.

**Therefore this is not a prose revision.** The plan rebuilds the method and experimental pipeline so every number in the resubmission comes from real, logged, non-circular experiments in this repo. Reported metrics will be whatever we measure — 0.961 is not a target. Decisions made: resubmit in the Oct–Jan window (AISTATS/ICML 2027); build true multi-agent MLCR rather than dropping the claim.

**Two verified root causes of the at-chance detector** (fix these first):
1. `eval/run_main.py` builds every DAG as a fixed-length sequential chain and `pge/graphon.py::_block_index` assigns blocks by node position → every trajectory has identical adjacency/blocks; the likelihood only varies through `|score|` (sign discarded), so confident-correct and confident-hallucinated nodes contribute identically.
2. Behavior signals in both benchmarks are truth-matches against the same ground truth that defines the eval label → circular evaluation.

**Standing rules:** eval labels may use ground truth; detector inputs may not be deterministic functions of it (enforced by a mechanical `assert_non_circular` check). Old code paths (positional blocks, `bayesian_score`, sequential edges) are kept as honest ablation arms, not deleted. Decouple LLM trajectory generation (paid, cached to JSONL) from detection (free, offline) so ablations/baselines are cheap.

## Reviewer-issue → plan mapping

| Issue (source) | Addressed by |
|---|---|
| Wrong Lemma 4.3, no corrected theorem (AC, Qg2c) | Phase 1: formal conservative-report lemma implemented in `scoring.py` with theory-derived revisitation threshold; unit-tested monotonicity/inverse |
| 96.4% multi-agent claim with no experiments (AC, Qg2c) | Phase 4: real k-agent MLCR + conflict-resolution experiments with vote baselines |
| Graphon ablation conflated components (AC, ap7L) | `dag_logreg` arm = full PGE minus graphon (same scoring + revisitation); block/edge-source ablation matrix |
| A1 calibration assumption unrealistic (ap7L, 2AM6) | Wire in `calibration.py`; ECE/reliability diagrams before/after; robustness-to-noise curves |
| Missing baselines: semantic entropy, etc. (ap7L) | Phase 3: semantic entropy, SelfCheckGPT, CoVe, DAG-feature LogReg, GCN |
| Benchmark overfitting / held-out generalization (2AM6) | `e5_crossdomain` train-on-A/test-on-B matrix |
| Reproducibility: prompts, procedure (2AM6, M447) | `pge/prompts/*.txt` versioned; config-driven runs; RESULTS.md ledger; seeds |
| DAG size, construction errors, worked example (M447) | Real CoT→DAG extractor + human-validated accuracy; `e5_dagsize`; `e5_robustness`; rendered end-to-end example figure |
| User-burden accounting (Qg2c) | `e5_burden`: question-count accounting, intake vs per-task |
| Unfair/fabricated mitigation rate | `e5_intervention`: measured post-intervention accuracy delta |

## Target layout (new/changed files)

```
pge/trajectory.py            NEW  Trajectory/Turn dataclasses, JSONL IO, assert_non_circular
pge/scoring.py               REWRITE  asymmetric_score, optimal_report, implied_belief, revisitation_threshold (+keep bayesian_score as ablation)
pge/graphon.py               EXTEND  block_method={position,degree,spectral,feature}, signed likelihood, length-normalized, (total,structure,score) breakdown
pge/calibration.py           WIRE IN  temp scaling + ECE reporting (currently dead code)
pge/protocol.py              EXTEND  formula-based revisitation trigger, calibration hook
pge/extraction/cot_dag.py    NEW  LLM CoT→DAG extractor (cheap 8B/Haiku); validation.py vs human annotations
pge/prompts/*.txt            NEW  every prompt versioned on disk
pge/baselines/               NEW  base.py (TrajectoryScorer), semantic_entropy.py, selfcheckgpt.py, cove.py, dag_logreg.py, gcn.py (tiny numpy GCN), confidence.py
pge/benchmarks/dbbench.py    MODIFY  non-circular BehaviorExtractorV2 (self-consistency probes, schema/existence checks, resample agreement); easier subset + schema+sample-rows prompting to lift 13.9% accuracy to 40–60%
pge/benchmarks/synthetic_healthcare.py  MODIFY  v2: noisy (ε=0.25) partial feedback, 12 attributes w/ interactions, variable T
pge/benchmarks/taubench.py   NEW  third domain: τ-bench retail/airline (local simulator, programmatic labels). Backup: StableToolBench static traces, detection-only
pge/multiagent/              NEW  commitments.py, conflict.py, resolve.py (reuses existing mlcr_resolve; vote baselines)
pge/judges/validation.py     NEW  judge-vs-human Cohen's κ on 100 hand-annotated turns (accept if κ≥0.7)
experiments/config.py, run.py, configs/*.yaml  NEW  config-driven generate|score|eval stages, seed control, git-SHA-stamped results
eval/generate_trajectories.py, score_trajectories.py, make_figures.py  NEW
data/trajectories/<domain>/<model>_<seed>.jsonl  NEW cached artifacts
results/<exp>/<seed>/ + RESULTS.md  NEW append-only ledger (feeds the paper)
tests/  NEW test_asymmetric_scoring.py, test_extraction.py, test_baselines.py, test_multiagent.py
```

## Phases & decision gates

**Phase 0 (week 1) — infra + honest re-baseline.** `trajectory.py`, `experiments/` runner, and `e0_diagnosis` (zero LLM cost): demonstrate from saved `.npz` that adjacency/blocks are constant, graphon-LL ≈ mean |score|, and circular signals reproduce the 0.99 AUROC. Becomes the paper's motivation + honest baseline row. **Gate G0 (day 3):** diagnosis logged in RESULTS.md.

**Phase 1 (weeks 1–3) — core method fix + synthetic go/no-go.**
- `scoring.py`: `asymmetric_score(p̂,y)= αp̂ if y=+1 else −βp̂^γ`; `optimal_report(q)=[αq/(βγ(1−q))]^{1/(γ−1)}` (the corrected, conservative lemma — formally stated in the paper); `implied_belief` (its inverse, applied before temperature scaling); `revisitation_threshold(q_min,t)` so the trigger is derived, not tuned.
- `graphon.py`: feature-based block assignment (confidence bucket × signal sign), signed likelihood, length normalization.
- `extraction/cot_dag.py`: agents emit reasoning; extractor model produces claim nodes + dependency edges; 50–100 human-annotated trajectories → node/edge P/R appendix table. Heuristic edges kept as ablation.
- Healthcare v2 (non-circular noisy feedback) + trivial/confidence + `dag_logreg` baselines.
- **Gate G1 (end wk 2–3), `e1_synth_gate`** (n=1000, 3 seeds, ~$1): GO if best graphon variant ≥0.65 AUROC and beats confidence baselines by ≥0.05 (non-overlapping CIs, ≥2/3 seeds). CONDITIONAL if ≈ dag_logreg → detector-zoo framing. NO-GO → fallback framing, no 70B graphon spend.

**Phase 2 (weeks 3–5) — benchmarks.** dbbench v2 (non-circular signals; accuracy fix verified by 50-task pilot before scaling); τ-bench as third domain with judge validation. Corpus (budget-capped): ~600–800 trajectories per domain, **70B as the single main model everywhere**, 8B added only on healthcare (near-free) for the model-size comparison; k=3 resamples generated only for final-answer turns (needed by entropy/SelfCheck). Gen cost target ≤ $20. **Gate G2:** first real dbbench-v2 70B result decides framing A/B/C.

**Phase 3 (weeks 4–6, overlaps) — baselines**, all offline on cached trajectories: semantic entropy (k=3 resamples, entailment clustering via 8B), SelfCheckGPT (LLM-prompt variant, 8B checker), CoVe (test-split only, capped at ~$5 — sample the test split if needed), numpy GCN, dag_logreg.

**Phase 4 (weeks 6–8) — true multi-agent MLCR.** k=3 agents (70B, 8B, and 70B at temp 0.9 with a different prompt as the free third agent) commit on the same session; conflict = incompatible commitments at p≥0.7; resolution via merged DAG + existing `mlcr_resolve` with `op_score` from `implied_belief`. Report conflict-resolution accuracy vs majority-vote, confidence-weighted vote, best-single-agent + oracle; McNemar test; n≈300 sessions × 2 domains (~$5–8).

**Phase 5 (weeks 8–11) — full matrix, robustness, figures.** Named experiments: `e5_main` (all detectors × domains × models, AUROC+AUPRC, bootstrap CIs, ≥3 seeds), `e5_ablation` (graphon/scoring/revisitation/calibration/block_method/edge_source/B sweep), `e5_calibration` (ECE, reliability diagrams), `e5_crossdomain`, `e5_robustness` (confidence jitter, edge noise), `e5_dagsize`, `e5_burden`, `e5_intervention` (measured mitigation delta — replaces the unsupported 87.3%). `make_figures.py` regenerates every figure from `results/`; one real rendered end-to-end DAG example.

**Phase 6 — paper (week 12+, LaTeX lives outside this repo).** Write strictly against RESULTS.md. Includes: corrected Lemma formally stated + proof, Theory-to-Implementation map table, all notation defined, all tables referenced, impact statement, fair Total-UB accounting, full prompt appendix, per-domain and DAG-size breakdowns. Framings by gate outcome: **A** graphon works → original structure with honest numbers; **B** graphon ≈ others → contribution = commitment-elicitation protocol + theory + systematic detector study (graphon as one detector, honest negative sub-result); **C** structure loses to confidence baselines → lead with multi-agent MLCR + theory, detection as an empirical finding.

## Budget & verification

- **Hard API budget cap: $50 total.** Allocation: Phases 0–1 gate experiments ≤ $3; Phase 2 trajectory generation ≤ $20; Phase 3 baselines (CoVe + checker calls) ≤ $8; Phase 4 multi-agent ≤ $8; Phase 5 intervention/revisitation regeneration ≤ $8; reserve $3. Every generation run records its actual cost in RESULTS.md (the Together API returns token counts; current 200-session 70B runs cost ~$0.34, so this cap is workable). **Stop-gate at $45 cumulative** — no new paid runs past it without explicit sign-off; all ablations/detector comparisons run offline on cached trajectories, which is what makes the cap feasible.
- Verification: unit tests for scoring math (monotonicity, `implied_belief∘optimal_report=id`), extraction accuracy vs human annotations, `assert_non_circular` lint in the trajectory loader, every experiment reproducible from its YAML + seed with git SHA stamped, RESULTS.md as the single source of truth for any number that enters the paper. Run `pytest` + `e1_synth_gate` end-to-end as the pipeline smoke test.
