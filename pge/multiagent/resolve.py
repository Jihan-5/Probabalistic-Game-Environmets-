"""Conflict resolvers: voting baselines, best-single-agent, and MLCR.

Four resolution strategies over `Conflict`s:

1. `majority_vote` — plurality over ALL commitments for the attribute
   (ties broken by summed reported probability).
2. `confidence_weighted_vote` — value maximizing the summed IMPLIED BELIEF
   (`pge.scoring.implied_belief`) of its supporters; reports are de-biased
   before they are used as weights.
3. `BestSingleAgent` — fit/predict oracle-free baseline: pick the agent with
   the best accuracy on TRAINING conflicts, then defer to that agent's value
   at test time (no test labels touched).
4. `mlcr_resolution` — build a MERGED DAG over every agent's commitments
   (per-agent dependency + sequential edges, plus cross-agent edges joining
   nodes for the same attribute), then run `pge.mlcr.mlcr_resolve` with
   feasible ops = the union of proposed values per conflicted attribute and
   an implied-belief op-score prior. The resolved value for a conflict is the
   majority over that attribute's per-node resolved operations.

Merged-DAG node scores use a PEER-AGREEMENT signal (+1 if the node's value
matches the cross-agent plurality for its attribute, -1 otherwise) — no
ground truth — so the "feature" graphon blocks can separate confident
contradicted-by-peers commitments from consensual ones.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from pge.benchmarks.healthcare_v2 import ATTRIBUTES_V2
from pge.constraints import SimpleConstraints
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.graphon import GraphonEstimator
from pge.mlcr import mlcr_resolve
from pge.multiagent.commitments import Commitment
from pge.multiagent.conflict import Conflict
from pge.scoring import bayesian_score, implied_belief

_LOG_EPS = 1e-6


# ------------------------------------------------------------------- voting


def _plurality(pairs: list[tuple[str, float]]) -> str:
    """Most common value; ties broken by highest summed weight."""
    counts: Counter = Counter(v for v, _ in pairs)
    weight: dict[str, float] = defaultdict(float)
    for v, w in pairs:
        weight[v] += w
    return max(counts, key=lambda v: (counts[v], weight[v]))


def majority_vote(conflict: Conflict) -> str:
    """Plurality over ALL commitments; ties -> highest summed probability."""
    return _plurality([(c.value, c.probability) for c in conflict.commitments])


def confidence_weighted_vote(conflict: Conflict) -> str:
    """Value maximizing sum of implied_belief(p) over supporting agents."""
    belief: dict[str, float] = defaultdict(float)
    for c in conflict.commitments:
        belief[c.value] += implied_belief(c.probability)
    return max(belief, key=lambda v: belief[v])


# --------------------------------------------------------- best single agent


class BestSingleAgent:
    """Pick the single most accurate agent on TRAINING conflicts, defer to it.

    fit() is the only place ground truth enters; predict() only reads the
    chosen agent's committed value (majority fallback if it is absent),
    so there is no label leakage at test time.
    """

    def __init__(self) -> None:
        self.best_agent_id: Optional[str] = None
        self.train_accuracy: Optional[float] = None

    def fit(
        self,
        conflicts: list[Conflict],
        truth_fn: Callable[[Conflict], str],
    ) -> "BestSingleAgent":
        correct: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for conflict in conflicts:
            truth = truth_fn(conflict)
            for c in conflict.commitments:
                total[c.agent_id] += 1
                if c.value == truth:
                    correct[c.agent_id] += 1
        if not total:
            raise ValueError("no training conflicts")
        # Deterministic tie-break: sort by (-accuracy, agent_id).
        scored = sorted(
            ((correct[a] / total[a], a) for a in total),
            key=lambda t: (-t[0], t[1]),
        )
        self.train_accuracy, self.best_agent_id = scored[0]
        return self

    def predict(self, conflict: Conflict) -> str:
        if self.best_agent_id is None:
            raise RuntimeError("BestSingleAgent not fit; call fit() first")
        for c in conflict.commitments:
            if c.agent_id == self.best_agent_id:
                return c.value
        return majority_vote(conflict)  # chosen agent silent on this attr


# ------------------------------------------------------------ merged DAG


def build_merged_dag(
    schedule: list[str],
    commitments_by_attr: dict[str, list[Commitment]],
    dependencies: dict[str, tuple[list[str], list[str]]] | None = None,
) -> ChainOfThoughtDAG:
    """Merge every agent's commitments for one session into a single DAG.

    Nodes: agent-major order — agent k's node for schedule position t sits at
    index k*T + t, `operation` = the committed value. Edges (forward only):
      - per-agent `attr:` dependency edges + sequential backbone,
      - cross-agent edges between every pair of nodes for the SAME attribute
        (earlier index -> later index).
    Node score = bayesian_score(p, peer-agreement sign): +1 if the value
    matches the cross-agent plurality for the attribute, else -1.
    """
    deps_table = ATTRIBUTES_V2 if dependencies is None else dependencies
    T = len(schedule)
    first = commitments_by_attr[schedule[0]]
    n_agents = len(first)

    plurality = {
        a: _plurality([(c.value, c.probability) for c in commitments_by_attr[a]])
        for a in schedule
    }

    G = ChainOfThoughtDAG()
    pos = {a: t for t, a in enumerate(schedule)}
    for k in range(n_agents):
        base = k * T
        for t, attribute in enumerate(schedule):
            c = commitments_by_attr[attribute][k]
            sign = 1 if c.value == plurality[attribute] else -1
            G.add_node(
                GuessNode(
                    attribute=attribute,
                    value=c.value,
                    probability=c.probability,
                    score=bayesian_score(c.probability, sign),
                    turn=t,
                    operation=c.value,
                )
            )
            idx = base + t
            for dep in deps_table.get(attribute, ([], []))[1]:
                if dep.startswith("attr:") and dep[5:] in pos:
                    parent = base + pos[dep[5:]]
                    if parent < idx and (parent, idx) not in G.edges:
                        G.add_edge(parent, idx)
            if t > 0 and (idx - 1, idx) not in G.edges:
                G.add_edge(idx - 1, idx)

    # Cross-agent edges: same attribute, earlier index -> later index.
    for t in range(T):
        for k1 in range(n_agents):
            for k2 in range(k1 + 1, n_agents):
                G.add_edge(k1 * T + t, k2 * T + t)
    return G


# --------------------------------------------------------------------- MLCR


def mlcr_resolution(
    G: ChainOfThoughtDAG,
    conflicts: list[Conflict],
    graphon: GraphonEstimator,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> dict[str, str]:
    """Resolve conflicts on a merged DAG via MLCR (paper Algorithm 4).

    Feasible ops for nodes of a conflicted attribute = union of the values
    proposed for it; every other node is pinned to its own proposal.
    op_score prior for node i proposing v with report p (k candidates):
        log(implied_belief(p) + eps)              for v,
        log((1 - implied_belief(p))/(k-1) + eps)  for each other candidate.
    Returns dict conflicted attribute -> resolved value (majority over that
    attribute's per-node resolved operations; ties -> summed implied belief
    of the original supporters).
    """
    proposals = {
        cf.attribute: sorted({c.value for c in cf.commitments}) for cf in conflicts
    }

    candidates: list[list[str]] = []
    op_score: dict[int, dict[str, float]] = {}
    for i, node in enumerate(G.nodes):
        cands = proposals.get(node.attribute, [str(node.operation)])
        candidates.append(list(cands))
        k = len(cands)
        if k <= 1:
            continue
        b = implied_belief(node.probability)
        op_score[i] = {
            op: float(
                np.log(b + _LOG_EPS)
                if op == node.operation
                else np.log((1.0 - b) / (k - 1) + _LOG_EPS)
            )
            for op in cands
        }

    resolved_ops = mlcr_resolve(
        G,
        graphon,
        SimpleConstraints(candidates=candidates),
        op_score=op_score,
        max_iter=max_iter,
        tol=tol,
    )

    resolved: dict[str, str] = {}
    for cf in conflicts:
        pairs = [
            (str(resolved_ops[i]), implied_belief(node.probability))
            for i, node in enumerate(G.nodes)
            if node.attribute == cf.attribute
        ]
        resolved[cf.attribute] = _plurality(pairs)
    return resolved


# --------------------------------------------- reliability-weighted MLCR (v2)


def _conf_bucket(p: float) -> int:
    return 0 if p < 0.5 else 1 if p < 0.75 else 2


class ReliabilityMLCR:
    """Reliability-weighted maximum-likelihood resolution.

    The plain graphon MLCR degenerates to majority vote on commitment
    conflicts (the graphon term is op-independent — see e4 summary). This
    resolver keeps the maximum-likelihood framing but learns the noise model
    the likelihood actually needs: per-BLOCK commitment reliability, where a
    block is (confidence bucket x cascade indicator). The cascade indicator
    marks agents whose commitments on the attribute's DEPENDENCY attributes
    disagree with the current consensus — exactly the confident-but-off-track
    coalition that fools plain voting.

    fit: on TRAIN conflicts, count per-block correctness (Laplace-smoothed) —
    the only place ground truth enters.
    predict: MAP under independent block-noise voters — each supporter adds
    log(acc_b / (1 - acc_b)) to its value; iterate so RESOLVED parent values
    update children's cascade indicators (the "consistent" part), 2 rounds.
    """

    def __init__(self, n_rounds: int = 2, laplace: float = 2.0):
        self.n_rounds = n_rounds
        self.laplace = laplace
        self.acc: dict[tuple[int, int], float] = {}
        self._deps_table = ATTRIBUTES_V2

    # ---- block computation

    def _parents(self, attribute: str) -> list[str]:
        return [
            d[5:] for d in self._deps_table.get(attribute, ([], []))[1]
            if d.startswith("attr:")
        ]

    def _cascade_bit(
        self,
        agent_id: str,
        attribute: str,
        commitments_by_attr: dict[str, list[Commitment]],
        parent_consensus: dict[str, str],
    ) -> int:
        """1 = on-track (agent agrees with consensus on parent attrs), 0 = off."""
        parents = [p for p in self._parents(attribute) if p in commitments_by_attr]
        if not parents:
            return 1
        agree = 0
        seen = 0
        for p in parents:
            mine = next(
                (c.value for c in commitments_by_attr[p] if c.agent_id == agent_id),
                None,
            )
            if mine is None or p not in parent_consensus:
                continue
            seen += 1
            agree += int(mine == parent_consensus[p])
        if seen == 0:
            return 1
        return 1 if agree * 2 >= seen else 0

    def _block(self, c: Commitment, cascade_bit: int) -> tuple[int, int]:
        return (_conf_bucket(c.probability), cascade_bit)

    @staticmethod
    def _consensus(commitments_by_attr: dict[str, list[Commitment]]) -> dict[str, str]:
        return {
            a: _plurality([(c.value, c.probability) for c in cs])
            for a, cs in commitments_by_attr.items()
        }

    # ---- supervised fit on train conflicts

    def fit(
        self,
        train_items: list[tuple[Conflict, dict[str, list[Commitment]], str]],
    ) -> "ReliabilityMLCR":
        """train_items: (conflict, full session commitments_by_attr, truth)."""
        correct: dict[tuple[int, int], float] = defaultdict(float)
        total: dict[tuple[int, int], float] = defaultdict(float)
        for cf, by_attr, truth in train_items:
            consensus = self._consensus(by_attr)
            for c in cf.commitments:
                bit = self._cascade_bit(c.agent_id, cf.attribute, by_attr, consensus)
                b = self._block(c, bit)
                total[b] += 1.0
                correct[b] += float(c.value == truth)
        self.acc = {
            b: (correct[b] + self.laplace * 0.5) / (total[b] + self.laplace)
            for b in total
        }
        return self

    # ---- MAP resolution with consistency iteration

    def _weight(self, block: tuple[int, int]) -> float:
        """Log-odds of block reliability, floored at 0.

        The pure MAP weight log(a/(1-a)) goes negative for a < 0.5, turning an
        unreliable supporter into a vote AGAINST its own value. With
        cascade-correlated errors that inversion misfires (measured: 0.592 vs
        0.662 majority, p=0.003 in the wrong direction), so unreliable blocks
        are ignored instead of inverted.
        """
        a = self.acc.get(block, 0.5)
        a = min(max(a, 0.02), 0.98)
        return max(float(np.log(a / (1.0 - a))), 0.0)

    def resolve_session(
        self,
        conflicts: list[Conflict],
        commitments_by_attr: dict[str, list[Commitment]],
    ) -> dict[str, str]:
        if not self.acc:
            raise RuntimeError("ReliabilityMLCR not fit; call fit() first")
        consensus = self._consensus(commitments_by_attr)
        resolved = {cf.attribute: consensus[cf.attribute] for cf in conflicts}
        for _ in range(self.n_rounds):
            # Parents' resolved values feed children's cascade indicators.
            ctx = dict(consensus)
            ctx.update(resolved)
            new_resolved = {}
            for cf in conflicts:
                weight: dict[str, float] = defaultdict(float)
                belief: dict[str, float] = defaultdict(float)
                for c in cf.commitments:
                    bit = self._cascade_bit(
                        c.agent_id, cf.attribute, commitments_by_attr, ctx
                    )
                    weight[c.value] += self._weight(self._block(c, bit))
                    belief[c.value] += implied_belief(c.probability)
                new_resolved[cf.attribute] = max(
                    weight, key=lambda v: (weight[v], belief[v])
                )
            if new_resolved == resolved:
                resolved = new_resolved
                break
            resolved = new_resolved
        return resolved
