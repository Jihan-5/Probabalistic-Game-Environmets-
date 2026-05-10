"""End-to-end PGE protocol (paper Algorithm 1).

Composes:
  1. one-time categorical setup,
  2. interactive guessing + behavioral scoring,
  3. graphon-based hallucination flag,
  4. MLCR resolution if hallucinated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from pge.agent import Agent, Guess
from pge.behavior import BehaviorExtractor
from pge.constraints import Constraints
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator
from pge.mlcr import mlcr_resolve
from pge.scoring import bayesian_score


@dataclass
class PGEResult:
    dag: ChainOfThoughtDAG
    log_likelihood: float
    hallucinated: bool
    output_operations: Optional[list]  # MLCR-resolved if hallucinated, else None


class PGEProtocol:
    """Algorithm 1 orchestrator.

    Parameters
    ----------
    setup_questions : list[str]
        Question identifiers; the user-supplied `setup_answers` mapping must
        contain a value for each.
    agent : Agent
    behavior_extractor : BehaviorExtractor
    graphon : GraphonEstimator (must already be fit)
    threshold_tau : float
        Hallucination flag if log-likelihood < tau.
    constraints : Constraints, optional
        Required only if MLCR resolution is desired.
    edge_window : int
        How many previous nodes each new node connects to (sequential dependency
        edges, paper Section 3.3). Default 1 = pure chain; larger values create
        denser DAGs.
    """

    def __init__(
        self,
        setup_questions: Sequence[str],
        agent: Agent,
        behavior_extractor: BehaviorExtractor,
        graphon: Optional[GraphonEstimator] = None,
        threshold_tau: Optional[float] = None,
        constraints: Optional[Constraints] = None,
        edge_window: int = 1,
        score_gated_edges: bool = False,
    ):
        if graphon is not None and graphon.W is None:
            raise ValueError("if a graphon is supplied it must be fit first")
        self.setup_questions = list(setup_questions)
        self.agent = agent
        self.behavior_extractor = behavior_extractor
        self.graphon = graphon
        self.threshold_tau = None if threshold_tau is None else float(threshold_tau)
        self.constraints = constraints
        self.edge_window = int(edge_window)
        self.score_gated_edges = bool(score_gated_edges)

    # ------------------------------------------------------ DAG construction

    def build_dag(
        self,
        setup_answers: Mapping[str, object],
        user_state: Mapping[str, object],
        T: int,
    ) -> ChainOfThoughtDAG:
        """Run the per-turn loop and return the chain-of-thought DAG.

        Does not require a fitted graphon — useful for collecting training
        data before fitting.

        Edge construction: each new node connects to up to `edge_window`
        immediately preceding nodes. If `score_gated_edges` is True, an edge
        is dropped whenever the parent's score is non-positive — modeling a
        broken reasoning chain when prior behavioral evidence contradicts
        the agent.
        """
        missing = [q for q in self.setup_questions if q not in setup_answers]
        if missing:
            raise ValueError(f"setup_answers missing keys: {missing}")

        G = ChainOfThoughtDAG()
        for t in range(T):
            guess: Guess = self.agent.guess(t, setup_answers, G)
            b_t = self.behavior_extractor.extract(guess, user_state, t)
            s_t = bayesian_score(guess.probability, b_t)
            idx = G.add_node(
                GuessNode(
                    attribute=guess.attribute,
                    value=guess.value,
                    probability=guess.probability,
                    score=s_t,
                    turn=t,
                )
            )
            for k in range(1, self.edge_window + 1):
                parent = idx - k
                if parent < 0:
                    continue
                if self.score_gated_edges and G.nodes[parent].score <= 0:
                    continue
                G.add_edge(parent, idx)
        return G

    def run(
        self,
        setup_answers: Mapping[str, object],
        user_state: Mapping[str, object],
        T: int,
    ) -> PGEResult:
        """Run a single multi-turn session end-to-end (Algorithm 1).

        Requires a fitted graphon and threshold tau (set in __init__).
        """
        if self.graphon is None or self.threshold_tau is None:
            raise RuntimeError(
                "PGEProtocol.run() requires both `graphon` and `threshold_tau`; "
                "use `build_dag()` for training-data generation."
            )
        G = self.build_dag(setup_answers, user_state, T)

        ll = self.graphon.log_likelihood(G)
        hallucinated = ll < self.threshold_tau
        resolved = None
        if hallucinated and self.constraints is not None:
            for node in G.nodes:
                if node.operation is None:
                    node.operation = node.value
            resolved = mlcr_resolve(G, self.graphon, self.constraints)

        return PGEResult(
            dag=G,
            log_likelihood=ll,
            hallucinated=hallucinated,
            output_operations=resolved,
        )

    # -------------------------------------------------------- threshold tuning

    @staticmethod
    def calibrate_tau(
        clean_dags: Iterable[ChainOfThoughtDAG],
        graphon: GraphonEstimator,
        fpr: float = 0.05,
    ) -> float:
        """Pick tau so that ~fpr fraction of clean DAGs fall below it.

        Paper §3.3: "threshold tau is calibrated on validation data for 5% false
        positive rate."
        """
        lls = np.array(
            [graphon.log_likelihood(G) for G in clean_dags], dtype=float
        )
        if lls.size == 0:
            raise ValueError("need at least one clean DAG to calibrate tau")
        return float(np.quantile(lls, fpr))
