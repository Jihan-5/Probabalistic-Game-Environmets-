"""Score a new healthcare corpus with the e6 protocol + ChannelPosterior ($0).

Thin wrapper around eval.e6_posterior.main (same splits, seeds, detectors,
paired bootstrap), adding pge_posterior_channel (V6, general feedback
channel) to the detector list, and registering the healthcare_llmpatient DAG
builder at RUNTIME (eval/score_trajectories.py is not edited). Works for
healthcare_v2 corpora from any model and for healthcare_llmpatient corpora.

Usage: python3 -m eval.paid.score_new_corpus <jsonl> [...] [--dag] > results/<exp>/<name>.txt
"""

from __future__ import annotations

import sys

import eval.e6_posterior as e6
import eval.score_trajectories as st
from pge.baselines.posterior import ChannelPosterior
from pge.benchmarks import healthcare_llmpatient

st.DAG_BUILDERS.setdefault(healthcare_llmpatient.DOMAIN, healthcare_llmpatient.dag_from_trajectory)

_base_make_detectors = e6.make_detectors


def make_detectors(domain):
    return _base_make_detectors(domain) + [ChannelPosterior()]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not [a for a in argv if not a.startswith("--")]:
        raise SystemExit(__doc__)
    e6.make_detectors = make_detectors
    sys.argv = [sys.argv[0], "healthcare", *argv]
    e6.main()


if __name__ == "__main__":
    main()
