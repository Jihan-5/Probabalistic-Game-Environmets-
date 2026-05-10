"""Probabilistic Game Environments (PGE).

Reference implementation of the four core algorithms from
Mankani, Jaiswal & Farooque, *Probabilistic Game Environments for Scalable
Hallucination Detection in Interactive Multi-Agent LLMs* (ICML 2026).

Public API:
    ChainOfThoughtDAG, GuessNode  - chain-of-thought DAG (Section 3.3)
    bayesian_score, cumulative_evidence  - Algorithm 2
    GraphonEstimator, gmixup  - Algorithm 3 / Eq. 6
    mlcr_resolve  - Algorithm 4
    PGEProtocol  - Algorithm 1
    Agent, StubAgent  - agent interface for the loop
    BehaviorExtractor, SyntheticBehaviorExtractor  - implicit-signal extraction
    Constraints, SimpleConstraints  - hard-constraint set for MLCR
"""

from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.scoring import bayesian_score, cumulative_evidence
from pge.graphon import GraphonEstimator, gmixup
from pge.mlcr import mlcr_resolve
from pge.protocol import PGEProtocol
from pge.agent import Agent, StubAgent
from pge.behavior import BehaviorExtractor, SyntheticBehaviorExtractor
from pge.constraints import Constraints, SimpleConstraints

__all__ = [
    "ChainOfThoughtDAG",
    "GuessNode",
    "bayesian_score",
    "cumulative_evidence",
    "GraphonEstimator",
    "gmixup",
    "mlcr_resolve",
    "PGEProtocol",
    "Agent",
    "StubAgent",
    "BehaviorExtractor",
    "SyntheticBehaviorExtractor",
    "Constraints",
    "SimpleConstraints",
]
