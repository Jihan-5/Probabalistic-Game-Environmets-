# Probabilistic Game Environments (PGE)

Reference implementation of the four core algorithms from
**Mankani, Jaiswal & Farooque, *Probabilistic Game Environments for Scalable
Hallucination Detection in Interactive Multi-Agent LLMs* (ICML 2026).**

The full paper proposes:

- **One-time categorical setup** (3–5 questions) instead of repeated detailed
  questioning.
- **Probabilistic guessing** by the agent with calibrated uncertainty.
- **Implicit behavioral scoring** of those guesses against the user's natural
  actions (no explicit ratings).
- **Sparse graphon-based detection** over chain-of-thought DAGs, with
  Maximum-Likelihood Consistent Resolution (MLCR) when conflicts arise.

This repo implements those four algorithms as a small NumPy/SciPy library plus
a synthetic toy demo. It is intended as a clean, extensible starting point —
not a full reproduction of the AgentBench / Health-LLM / ToolBench evaluation
in the paper.

## Layout

| Module | Paper reference |
|---|---|
| [`pge/dag.py`](pge/dag.py) | §3.3 — chain-of-thought DAG |
| [`pge/scoring.py`](pge/scoring.py) | Algorithm 2, Eqs. 1–2 — Bayesian behavioral scoring |
| [`pge/graphon.py`](pge/graphon.py) | Algorithm 3, Eqs. 3–6 — sparse graphon estimation, likelihood, G-Mixup |
| [`pge/mlcr.py`](pge/mlcr.py) | Algorithm 4, Eq. 7 — MLCR via EM |
| [`pge/protocol.py`](pge/protocol.py) | Algorithm 1 — end-to-end orchestrator |
| [`pge/agent.py`](pge/agent.py) | §3.1 — `Agent` ABC + `StubAgent` |
| [`pge/behavior.py`](pge/behavior.py) | §3.2 — `BehaviorExtractor` ABC |
| [`pge/constraints.py`](pge/constraints.py) | §3.4 — hard-constraint set Ω_C |
| [`examples/healthcare_demo.py`](examples/healthcare_demo.py) | Synthetic toy demo |
| [`tests/`](tests/) | Unit tests for each module + end-to-end pipeline |

## Install

```bash
pip install -e .
pip install -e ".[dev]"   # for pytest
```

Requires Python ≥ 3.9, NumPy ≥ 1.23, SciPy ≥ 1.10.

## Quickstart

```python
from pge import (
    ChainOfThoughtDAG, GuessNode,
    GraphonEstimator,
    bayesian_score,
)

# Build a single chain-of-thought DAG.
G = ChainOfThoughtDAG()
for t in range(5):
    p = 0.7
    s = bayesian_score(p, behavior_signal=+1)  # b_t in {-1, +1}
    idx = G.add_node(GuessNode(
        attribute="risk_level", value="moderate",
        probability=p, score=s, turn=t,
    ))
    if idx > 0:
        G.add_edge(idx - 1, idx)

# Fit a graphon on a batch and score a new DAG.
estimator = GraphonEstimator(B=10).fit([G, G, G])
print(estimator.log_likelihood(G))
```

The full pipeline (Algorithm 1) lives in `PGEProtocol`:

```python
from pge import PGEProtocol, StubAgent, SyntheticBehaviorExtractor

protocol = PGEProtocol(
    setup_questions=["risk", "horizon", "goal"],
    agent=my_agent,                # must implement pge.Agent
    behavior_extractor=my_extractor,
    graphon=fitted_graphon,
    threshold_tau=tau,             # see PGEProtocol.calibrate_tau
)
result = protocol.run(setup_answers, user_state, T=10)
print(result.log_likelihood, result.hallucinated)
```

## Run the demo

```bash
python examples/healthcare_demo.py --n-train 500 --n-test 100 --turns 6
```

The demo simulates a healthcare-scheduling task with 3 categorical setup
questions, fits a sparse graphon on clean sessions, calibrates τ at 5% FPR,
then evaluates hallucination detection on a held-out split where the
agent receives mismatched setup info. AUROC is reported at the end.

## Tests

```bash
pytest tests/ -v
```

Covers: scoring math (range, monotonicity, clipping), DAG operations,
graphon fit / likelihood / G-mixup, MLCR convergence and constraint
respect, and an end-to-end pipeline test.

## Demo results

Running `examples/healthcare_demo.py` across 5 seeds (`{42, 123, 456, 789, 1024}`)
with 500 training sessions and 200 + 200 clean/hallucinating test sessions per
seed:

| Metric | Mean ± std |
|---|---|
| AUROC | **0.954 ± 0.016** |
| Flag rate, clean (target ≤ 5% FPR) | 5.8% |
| Flag rate, hallucinating | 91.5% |
| Clean log-likelihood | −0.087 |
| Hallucinating log-likelihood | −8.72 |

Per-seed AUROC: 0.960 / 0.977 / 0.944 / 0.956 / 0.931.

**Caveat — these are toy numbers.** The synthetic distribution is easy by
construction: under `score_gated_edges=True`, hallucinating sessions
collapse to nearly-empty DAGs (parents with negative score are dropped),
which the graphon trivially distinguishes from the dense chains of clean
sessions. The paper's 0.961 ± 0.012 AUROC is on real LLM-generated
hallucinations against AgentBench / Health-LLM / ToolBench — a much
harder problem that this reference implementation does not attempt.
The demo is here to confirm the pipeline is wired correctly, not to
validate the paper's empirical claims.

## Plugging in a real LLM

Replace `StubAgent` with an `Agent` subclass whose `guess()` method calls a real
LLM and returns a `Guess(attribute, value, probability)`. Use temperature
scaling (paper §3.1) on a held-out validation set to calibrate the probability.
The rest of the pipeline (scoring, graphon, MLCR) is LLM-agnostic.

## Limitations of this implementation

- **No real LLM integration.** `StubAgent` is rule-based.
- **No benchmark harness.** AgentBench / Health-LLM / ToolBench wiring is left
  to follow-on work.
- **No theorem-validation experiments.** Theorems 4.1–4.4 are implemented in
  the math (motif preservation under mixup, sparse-regime estimation, scoring
  noise bound, sample complexity); empirically validating their bounds is out
  of scope for this reference repo.
- **No meta-learning prior** (paper §6.6).

## License

Apache 2.0 (per paper §7).
