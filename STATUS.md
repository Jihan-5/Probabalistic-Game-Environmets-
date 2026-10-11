# PGE Rebuild — Live Status

*Maintained autonomously by Claude. Updated every working session. Numbers-of-record live in `results/RESULTS.md`; this file is the narrative.*

**Target: AISTATS 2027 — abstract Sept 29, full paper Oct 6 (AoE).** Hard API budget: $50.

## The method in one paragraph (what/why/how)

Interactive LLM agents fabricate facts about *the user* (risk tolerance, allergy severity, urgency) when context is missing, then act on them confidently — a failure mode we call **user-specific hallucination**. It cannot be caught by fact-checking because the truth is private to the user. **PGE** makes the agent expose this failure surface: at every turn the agent must state each assumption as an explicit probabilistic commitment ("needs_contrast: yes, p=0.85"). The environment scores commitments against *implicit, noisy* behavioral feedback (never the ground truth itself) with an **asymmetric scoring rule** — confident errors cost super-linearly more (−βp̂^γ) than cautious ones earn (+αp̂). The corrected theory (conservative-report lemma) shows the optimal strategy under this rule under-reports confidence in a precisely invertible way, so raw reports can be de-biased into beliefs and intervention thresholds derived rather than tuned. Commitments link into a session-level reasoning DAG, and detectors score whole sessions: confidence marginals, cascade statistics (connected clusters of contradicted high-confidence commitments), supervised graph features, and graphon likelihood. Flagged sessions trigger revisitation or a clarifying question; in multi-agent deployments, conflicting commitments are resolved by maximum-likelihood consistency (MLCR). The paper's claim is no longer "graphon detects hallucination" (our own Gate G1 killed that: 0.61–0.67 AUROC vs 0.78–0.82 for simpler detectors) — it is the **protocol + theory + an honest systematic detector study** on real LLM trajectories.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0. Infra + honest re-baseline | ✅ done | e0_diagnosis: old pipeline invalidated (structure-blind + circular), $0 |
| 1. Core fix + Gate G1 | ✅ done | Corrected scoring theory implemented+tested; G1 = NO-GO for graphon-as-flagship → framing B |
| 2. Benchmarks + corpus | ✅ done | **1,272 real LLM trajectories**: dbbench (118 tasks × 3 seeds × 70B + 118 × 8B) + healthcare (600 × 70B w/ resamples + 200 × 8B). τ-bench cut (sprint). |
| 3. Baselines | ✅ done | semantic entropy, SelfCheckGPT, CoVe (injectable LLM), numpy GCN — 23 tests |
| 4. Multi-agent MLCR | ✅ done | Plain MLCR ≡ majority (degeneracy found+documented); **ReliabilityMLCR beats majority 0.685 vs 0.662, p=0.047** (784 conflicts); oracle 0.996 |
| 5. Experiment matrix + figures | ✅ done | Main tables (per-domain, per-model), cross-domain, robustness, calibration/ECE, DAG-size stratification, **intervention**, 6 figures. All experiments logged in results/RESULTS.md |
| 6. Paper | ⬜ ready to write | LaTeX outside repo; every table/figure/number now exists in results/ and figures/ |

## The paper's result story (all real, all logged)

1. **Detection tables (1,272 trajectories, 2 domains × 2 models):** behavioral-evidence detectors lead everywhere (dbbench: selfcheckgpt 0.712 / dag_logreg 0.681; healthcare: highconf_contradicted 0.711 / dag_logreg 0.705 / gcn 0.679; 8B: dag_logreg 0.717). Graphon likelihood ≤ chance in every real setting — reported as an honest negative with the e0/e1 diagnosis explaining why.
2. **Headline contrast:** sampling-consistency detectors (SelfCheckGPT 0.504, semantic entropy 0.499) are BLIND to user-specific hallucination on healthcare — the model repeats the same fabrication on resampling (3.5% vs 4.3% disagreement, clean vs hallucinating) — while they work on dbbench's computational errors (0.712/0.673). Direct evidence that user-specific hallucination is a distinct failure mode, which is the paper's thesis. Nuance (e2_healthcare_8b): the contrast is scale-dependent — for the 8B agent selfcheckgpt recovers signal (0.669) while behavioral stats weaken (0.596), i.e. small models are less self-consistent when fabricating.
3. **Corrected theory:** conservative-report lemma with exact regions (conservative iff q < 1 − α/(βγ); saturation at βγ/(α+βγ)), invertible de-biasing, formula-derived revisitation threshold; unit-tested.
4. **Intervention (honest mitigation number):** flag top-30% risk → revisitation resolves **62–69%** of flagged hallucinating sessions (two independent reruns: 69.0% and 62.1%), 4.0% collateral, **−9.4 to −10.6pp absolute** hallucination on test.
5. **Multi-agent:** ReliabilityMLCR 0.685 vs majority 0.662 (p=0.047, 784 conflicts); oracle 0.996 ⇒ clarification is the future-work headline.
6. **Generalization & robustness:** supervised detectors do NOT transfer across domains (≈ chance) while zero-shot evidence signals do; all detectors degrade gracefully under noise (dag_logreg 0.735 → 0.685 worst).
7. **Calibration:** temperature scaling improves ECE on every seed/domain (e.g., 0.189 → 0.120).

## Key decisions log

- **2026-09-12** — $50 hard API cap (user). Multi-agent MLCR: build for real (user).
- **2026-09-12** — Gate G0: pre-rebuild results invalidated; every number must come from new non-circular runs.
- **2026-09-13** — Gate G1 NO-GO: graphon likelihood loses to simpler detectors on its best-case synthetic environment. Paper framing B adopted: protocol + theory + detector study; graphon reported honestly as negative.
- **2026-09-13** — AISTATS 2027 confirmed target (abstract 9/29, paper 10/6). τ-bench third domain demoted to stretch goal; volume comes from healthcare-v2 (LLM) + dbbench-v2 (all 120 real tasks × models × seeds, task-level splits).
- **2026-09-13** — dbbench corpus is capped at ~120 SELECT-style tasks in the public data; widened task types (8 types, ≤12 cols) and made the answer probe type-aware.

## Open questions for Jihan (answer when you can; none block progress)

1. **Author list**: same as ICML submission (Mankani, Farooque — pyproject also lists Jaiswal)? Needed for the AISTATS reciprocal-reviewer nomination (one author must review).
2. **Human annotation**: plan needs ~2–4 hours of your time labeling ~100 turns (judge validation) + ~50 extraction DAGs. When can you schedule it? (Needed before the paper claims judge/extraction validity; not needed for the main detector tables.)
3. LaTeX: do you have the ICML source to port, or start fresh from the AISTATS template?

## Budget

Spent: **$7.96 of $50** (full breakdown in results/RESULTS.md). All planned experiments complete — remaining budget is reserve for paper-revision follow-ups. Stop-gate $45.

## What's left (needs Jihan)

1. **Paper writing** (LaTeX, outside this repo) — every number/table/figure is ready in `results/` and `figures/`. Suggested structure follows the 7-point result story above.
2. **Annotation (~2–4 hrs)**: ~100 turns for judge validation is now OPTIONAL (labels are programmatic in both domains — no LLM judge in the final pipeline), but a 50-trajectory human audit of the healthcare eval labels would strengthen the methods section.
3. Answers to the 3 open questions above (authors/reciprocal reviewer, annotation timing, LaTeX source).
