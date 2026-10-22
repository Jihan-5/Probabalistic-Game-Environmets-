# USHER / Probabilistic Game Environments (PGE)

Detecting **user-specific hallucination** in interactive LLM agents: confident
assertions about facts of the current user (an allergy's severity, days since
a treatment, a risk tolerance) that the agent was never given. The truth is
private to the user, so these errors cannot be fact-checked from the agent's
context, and resampling cannot separate users who induce the same context
(Theorem 1 in the paper; `eval/e9_consistency_bound.py` for the empirical
check).

In a USHER session the agent states every assumption as an explicit
commitment, `[attribute]: value, p=confidence`, and the user returns an
implicit reaction (contradict / no reaction / accept). Detectors score whole
sessions and individual commitments; high-risk commitments trigger a targeted
confirmation question.

## What this repository contains

- **Registered analysis plan** (`results/PREREG_confirmatory.md`) committed
  before any confirmatory data existed, plus seven recorded amendments
  (`results/PREREG_commits.md`). The registered primary posterior (V6) and
  question policy **failed their comparisons**; the intake-aware posterior
  (V7) and the revised policy were registered afterward as Amendments 6–7 and
  are reported as amended results, not as the original plan.
- **Confirmatory corpora** (`data/trajectories/`), real LLM agents against
  simulated users, protocol `v2fb`, seed 7 (never used in development):
  - `healthcare_confirm/`: simulated imaging clinic, 600 sessions per agent
    for Llama-3.3-70B, Llama-3.1-8B, Qwen2.5-72B, GPT-4o-mini, and
    Mistral-Small-3.1-24B. Reactions are the commitment's correctness bit
    flipped with probability 0.25; two attributes never react.
  - `travel_confirm/`: travel-booking domain (Amendment 4), 600 sessions
    each for GPT-4o-mini and Llama-3.1-8B.
  - `healthcare_llmpatient_v2/`: natural-language GPT-4o-mini patient
    (Amendment 5), 600 sessions; its pilot passed the registered
    contradiction-lift gate (2.69 ≥ 1.5). The v1 patient failed the gate
    (lift 1.06 / 1.21) and was stopped at 100-session pilots.
  - `healthcare_v2/`, `dbbench_v2/`: the development corpora (seed 42,
    protocol `v2`, corrupted feedback display — development evidence only).
- **Detectors** (`pge/baselines/`): sampling baselines (SelfCheckGPT,
  semantic entropy), CoVe and an LLM judge (`eval/paid/`), feedback
  statistics, context/intake logistic regressions, and the label-event
  posteriors in `posterior.py` — V6 (registered primary, general channel)
  and V7 (Amendment 6: intake-conditioned cells, learned label event).
- **Experiments** (`eval/`): e11–e22 are the confirmatory and amended
  analyses; e0–e9 are the development-era studies and remain as the record.
- **Ledger** (`results/RESULTS.md`): append-only, every run including losers,
  discarded variants, deviations, and corrections, each with its artifact.
- **Paper tooling** (`paper/`): `make_numbers.py` writes every number in the
  manuscript from the result artifacts; `fig_*.py` build the figures.

Every number reported from this repository should trace to a row of
`results/RESULTS.md`.

## Main findings (confirmatory; see the ledger for intervals and caveats)

1. **Sampling-consistency detectors are near chance on the user-specific
   label L_U**: pooled over five healthcare agents, SelfCheckGPT 0.551 and
   semantic entropy 0.495 session AUROC; per-commitment resample-disagreement
   AUROC 0.501–0.555 (`eval/e11_confirmatory.py`). SelfCheckGPT reaches
   0.645 on the composite label L (which includes clinic-rule errors) and
   0.628 on travel Llama-8B, so the limitation is specific to user-private
   facts, not hallucination detection generally.
2. **The registered V6 posterior did not beat the supervised feedback
   classifier** (V6 − ctx-LR+F = −0.050). The feedback increment itself is
   real: reactions add +0.070 to V6 over its prior-only version and +0.061 to
   the classifier (H2/H2b, all five agents).
3. **The amended intake-aware posterior V7 reaches 0.825 pooled session
   AUROC on L_U** (0.902 on Llama-70B, held unseen at amendment time), beating
   V6 by +0.162 (H7) and intake-LR+F by +0.089 (H8); Mistral-24B is the
   non-significant exception. A logistic regression given V7's own
   intake-conditioned, schema-selected features ties it (+0.012 n.s.): the
   dependency **structure**, not the Bayesian estimator, explains the
   session-level gain (`eval/e16_v7_confirm.py`, `eval/e17_strong_baselines.py`).
   The travel domain replicates H7 (+0.091) and H8 (+0.050).
4. **Clarification (Amendment 7, one agent, 120 sessions)**: one V7-selected
   question leaves 0.367 wrong confident user-dependent commitments per
   session versus 0.583 for a random question (−0.217 [−0.325, −0.108]);
   the registered V6 picker had failed (−0.008 vs random)
   (`eval/e21_clarification_v7.py`).
5. **The main external-validity limit**: on the natural-language patient v2
   corpus, frozen V7 scores 0.719 and trails intake-LR+F by 0.060; adding
   reactions to the classifier is worth +0.005 (n.s.). The headline results
   characterize the designed noisy-bit channel, not incidental human
   feedback (`eval/e22_patient_v7.py`).

## Reproduce (offline; no API calls)

Requires Python >= 3.9, NumPy, SciPy (matplotlib for figures). All scoring
runs on the cached trajectories.

```bash
python3 -m eval.e11_confirmatory              # registered analysis, 5 agents + travel + patient v2 (--dev for development corpora)
python3 -m eval.e16_v7_confirm                # V7 (Amendment 6) on all confirmatory corpora
python3 -m eval.e17_strong_baselines          # tuned LR / GBM / structure-matched comparators
python3 -m eval.e18_joint_bootstrap           # joint-session bootstrap + Holm for the registered families
python3 -m eval.e19_percommit                 # per-commitment AUROC and calibration
python3 -m eval.e20_judge                     # LLM-judge scoring (reads cached judge outputs)
python3 -m eval.e21_clarification_v7          # H9 analysis of the closed-loop runs
python3 -m eval.e22_patient_v7                # frozen V7 on the patient v2 corpus
python3 -m eval.e15_regimes --confirm --draws 5   # feedback noise / coverage regimes
python3 -m eval.e9_consistency_bound          # per-commitment self-consistency evidence
python3 -m eval.theorem2_counterexample       # adaptive-agent counterexample to Theorem 2's premise
python3 paper/make_numbers.py                 # regenerate every number cited in the paper
```

Corpus generation and the closed-loop clarification runs call paid model APIs
(`eval/paid/`, `run_paid.sh`; plan and caps in `RUN_PAID.md` — historical
record, all steps have run). Keys go in `.env`; see `pge/config.py`. Hard
budget cap $50; every dollar is logged in the ledger.

## Rules the code enforces

- Evaluation labels may use ground truth; detector inputs may not be
  deterministic functions of it (`pge/trajectory.py: assert_non_circular`).
- Splits are by session/task id, so no session appears on both sides; three
  70/30 splits, AUROC averaged, paired bootstrap clustered by session.
- Per-turn truth is reconstructible from the generator seed for analysis and
  for the oracle controls (η̄, η_X, η_F); it never feeds a detector. The
  user-specific label `meta.eval_label_user` is truth-derived and
  detector-invisible.
- Protocols: `v2` (development) showed the agent a corrupted feedback
  display; `v2fb` (confirmatory) shows the raw reaction. Scripts that rebuild
  agent prompts read `meta.protocol` and reproduce the same display.

## License

Apache 2.0.
