"""Maximum-Likelihood Consistent Resolution — paper Algorithm 4 / Eq. 7.

When hallucination is flagged, multiple agents may have proposed conflicting
operations. MLCR selects the assignment that maximizes graphon likelihood
subject to hard constraints, via Expectation-Maximization:

  E-step: estimate each node's latent block position from its empirical
          adjacency to other nodes (paper Algorithm 4 line 6).
  M-step: for each node, replace its operation with the feasible op
          that maximizes per-node conditional log-likelihood under W.

Convergence: stop when ||O^(k+1) - O^(k)||_1 < tol or max_iter reached.
Theorem 4.5 guarantees geometric convergence under Lipschitz L with rho < 1.
"""

from __future__ import annotations

import numpy as np

from pge.constraints import Constraints
from pge.dag import ChainOfThoughtDAG
from pge.graphon import GraphonEstimator, _block_index, _W_EPS


def _operation_distance(a: list, b: list) -> float:
    """L1-style distance over an operation assignment.

    Operations may be arbitrary objects; we count positions where they differ.
    """
    return float(sum(0.0 if x == y else 1.0 for x, y in zip(a, b)))


def _node_loglik(
    node_idx: int,
    op: object,
    operations: list,
    A: np.ndarray,
    blocks: np.ndarray,
    W: np.ndarray,
    op_score: dict | None,
) -> float:
    """Per-node conditional log-likelihood used in the M-step.

    We score `op` at `node_idx` by:
      (a) the graphon log-likelihood of the node's edges under its current block,
      (b) plus an optional op-specific affinity term `op_score[node_idx][op]`
          which lets callers express how well an operation fits the node's
          observed signal (defaults to 0 for any unspecified entry).

    Splitting (a) and (b) keeps the EM cleanly separable: (a) is what changes
    when E-step updates blocks, (b) is what changes when M-step considers
    different ops.
    """
    n = A.shape[0]
    a_self = blocks[node_idx]
    ll = 0.0
    for j in range(n):
        if j == node_idx:
            continue
        a_other = blocks[j]
        # symmetric contribution; we don't double-count because each call covers
        # only the focal node.
        if node_idx < j:
            edge = A[node_idx, j]
            p = W[a_self, a_other]
        else:
            edge = A[j, node_idx]
            p = W[a_other, a_self]
        ll += edge * np.log(p) + (1 - edge) * np.log(1.0 - p)

    if op_score is not None:
        ll += float(op_score.get(node_idx, {}).get(op, 0.0))

    return ll


def mlcr_resolve(
    G: ChainOfThoughtDAG,
    graphon: GraphonEstimator,
    constraints: Constraints,
    op_score: dict | None = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> list:
    """Resolve conflicting operations via EM (paper Algorithm 4).

    Parameters
    ----------
    G : ChainOfThoughtDAG
        Reasoning trace whose node `operation` fields hold the initial
        proposed operations.
    graphon : GraphonEstimator
        Fitted graphon W.
    constraints : Constraints
        Hard-constraint set; M-step searches over `feasible_ops`.
    op_score : dict[int, dict[op, float]], optional
        Per-(node, op) affinity term added to the graphon log-lik in the M-step.
    max_iter : int
        Paper uses 20 (Algorithm 4 line 4).
    tol : float
        Paper uses 1e-4 (Algorithm 4 line 11).

    Returns
    -------
    list of operations, one per node, satisfying constraints.
    """
    if graphon.W is None:
        raise RuntimeError("graphon not fit")

    n = len(G)
    if n == 0:
        return []

    W = np.clip(graphon.W, _W_EPS, 1.0 - _W_EPS)
    B = graphon.B
    A = G.adjacency()
    A_sym = A + A.T  # for E-step empirical position

    operations = [node.operation for node in G.nodes]
    blocks = np.array([_block_index(i, n, B) for i in range(n)])

    for _ in range(max_iter):
        # ---- E-step: update latent positions from empirical adjacency.
        if n > 1:
            avg_adj = A_sym.sum(axis=1) / max(n - 1, 1)
            # Map [0, 1] avg adjacency to a block via the same convention used
            # for nodes: rank nodes by avg_adj, then assign block by rank.
            order = np.argsort(avg_adj)
            new_blocks = np.empty(n, dtype=int)
            for rank, idx in enumerate(order):
                new_blocks[idx] = _block_index(rank, n, B)
            blocks = new_blocks

        # ---- M-step: replace each op with the feasibly best.
        new_operations = list(operations)
        for i in range(n):
            feasible = constraints.feasible_ops(i, new_operations)
            if not feasible:
                continue
            best_op = new_operations[i]
            best_ll = -np.inf
            for op in feasible:
                trial = list(new_operations)
                trial[i] = op
                ll = _node_loglik(i, op, trial, A, blocks, W, op_score)
                if ll > best_ll:
                    best_ll = ll
                    best_op = op
            new_operations[i] = best_op

        delta = _operation_distance(new_operations, operations)
        operations = new_operations
        if delta < tol:
            break

    return operations
