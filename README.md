# Probabilistic Game Environments (PGE)

Detecting **user-specific hallucination** in interactive LLM agents: confident
assertions about facts of the current user (an allergy's severity, days since
a treatment, a risk tolerance) that the agent was never given. The truth is
private to the user, so these errors cannot be fact-checked.

In a PGE session the agent states every assumption as an explicit commitment,
`[attribute]: value, p=confidence`, and the environment returns implicit
feedback (contradict / no reaction / accept). Detectors score whole sessions;
flagged sessions trigger revisitation or a targeted confirmation question.

## What this repository contains

- **Cached trajectories** (`data/trajectories/`), 1,272 real LLM sessions:
  - `healthcare_v2/`: a simulated imaging clinic. The agent (Llama-3.3-70B,
    Llama-3.1-8B) is real; the patient is simulated and its feedback is the
    commitment's correctness bit flipped with probability 0.25.
  - `dbbench_v2/`: AgentBench database tasks with table-derived probes.
- **Detectors** (`pge/baselines/`): sampling (SelfCheckGPT, semantic entropy),
  feedback statistics, DAG features, GCN, graphon likelihood, and the
  label-event posteriors in `posterior.py`.
- **Experiments** (`eval/`), all offline on the cached trajectories.
- **Ledger** (`results/RESULTS.md`): every run, including variants that lost
  and results that were later discarded or corrected, with its artifact.

Every number reported from this repository should trace to a row of
`results/RESULTS.md`.

## Main findings (see the ledger for intervals and caveats)

1. Sampling-consistency detectors are near chance on user-specific
   hallucination with the 70B agent (SelfCheckGPT 0.504, semantic entropy 0.499
   session AUROC) while working on verifiable SQL tasks (0.712 / 0.673). On
   underdetermined attributes the 70B agent's resamples agree on 92.6% of
   commitments although 41% are wrong (`eval/e9_consistency_bound.py`).
2. The label-event posterior (`IndependentPosterior(by_value=True)`) is the
   strongest detector on the healthcare corpus: 0.775 vs 0.711 for the best
   baseline (70B). The 8B margin is smaller and less stable.
   `ChannelPosterior` frees the feedback channel and remains robust when
   feedback is one-sided.
3. Structural detection over reasoning DAGs (graphon likelihood, cascade
   models) does not beat per-commitment evidence. A naive (v1) pipeline appears
   to succeed only because of constant graph topology and a signal that is a
   match against the evaluation label (`eval/e0_diagnosis.py`).
4. Choosing which commitments to confirm with the user by posterior risk
   matches or beats asking every underdetermined attribute upfront, at a
   quarter of the questions (`eval/e8_clarification.py`, offline lower bound).

## Reproduce

Requires Python >= 3.9, NumPy, SciPy (matplotlib for figures). No API calls.

```bash
python3 -m eval.e6_posterior healthcare data/trajectories/healthcare_v2/70B_42.jsonl   # main detection table
python3 -m eval.e6_ceiling            # oracle ceiling (analysis only)
python3 -m eval.e7_ablation           # protocol ablation
python3 -m eval.e7_label_sensitivity  # alternative label definitions
python3 -m eval.e7_channel_misspec    # misspecified feedback channels
python3 -m eval.e7f_channel_posterior # general-channel posterior under those channels
python3 -m eval.e7_ceiling_noise      # detection vs feedback noise
python3 -m eval.e8_clarification --corpus 70B_42   # which commitments to confirm
python3 -m eval.e9_consistency_bound  # per-commitment self-consistency evidence
python3 -m eval.e10_paper_figures     # figures from result files
```

Trajectory generation (`eval/generate_*.py`) calls paid model APIs and needs
keys in the environment; see `pge/config.py`.

## Rules the code enforces

- Evaluation labels may use ground truth; detector inputs may not be
  deterministic functions of it (`pge/trajectory.py: assert_non_circular`).
- Splits are by task id, so a task never appears on both sides.
- Healthcare per-turn truth can be reconstructed from the generator seed for
  analysis (`eval/e6_ceiling.py: load_with_truth`); it never feeds a detector.

## License

Apache 2.0.
