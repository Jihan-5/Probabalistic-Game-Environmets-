"""Multi-agent conflict resolution (commitments -> conflicts -> resolvers).

Free synthetic multi-agent layer over healthcare_v2: heterogeneous agent
profiles commit independently, `find_conflicts` detects confident
disagreement, and `resolve` provides voting baselines, best-single-agent,
and graphon-based MLCR resolution on a merged cross-agent DAG.
"""

from pge.multiagent.commitments import Commitment, collect_commitments
from pge.multiagent.conflict import Conflict, find_conflicts
from pge.multiagent.resolve import (
    BestSingleAgent,
    ReliabilityMLCR,
    build_merged_dag,
    confidence_weighted_vote,
    majority_vote,
    mlcr_resolution,
)

__all__ = [
    "Commitment",
    "collect_commitments",
    "Conflict",
    "find_conflicts",
    "majority_vote",
    "confidence_weighted_vote",
    "BestSingleAgent",
    "build_merged_dag",
    "mlcr_resolution",
    "ReliabilityMLCR",
]
