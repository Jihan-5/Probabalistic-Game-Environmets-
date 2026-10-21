"""e22: the frozen V7 on the LLM-patient v2 corpus (EXPLORATORY).

    python3 -m eval.e22_patient_v7

Amendment 5 registers the LLM-patient v2 analysis (V6, V6-prior, ctx-LR and
ctx-LR+F on L_U and L; run by e11). Amendment 6 does not name this corpus, so
V7 on it is exploratory. Same splits, bootstrap and outputs as e16's block,
with the sampling baselines left out (this corpus has no resamples). V7 is the
frozen d32ea33 detector; nothing is tuned here.
Writes results/e22_patient_v7/{summary.md, numbers.json}.
"""

from __future__ import annotations

import json
import time

from eval import e16_v7_confirm as e16
from eval.e11_confirmatory import PATIENT, REPO
from eval.e12_exploratory import IntakeLR
from pge.baselines.context_lr import ContextLR
from pge.baselines.intake_posterior import v7
from pge.baselines.posterior import ChannelPosterior

OUT = REPO / "results" / "e22_patient_v7"
TAG = "70B LLM-patient v2 (profile)"


def factories(dom, lab):
    ea = dom.USER_DEPENDENT_SCHEMA if lab == "L_U" else None
    return [("V6", lambda: ChannelPosterior(event_attrs=ea)), ("V7", lambda: v7(dom, lab)),
            ("ctx_lr", lambda: ContextLR()), ("ctx_lr+fb", lambda: ContextLR(with_feedback=True)),
            ("intake_lr+fb", lambda: IntakeLR(True))]


def main():
    t0 = time.time()
    e16.factories = factories  # block() looks these up at call time
    e16.COMPS = [("H7", "V7", "V6"), ("H8", "V7", "intake_lr+fb"), ("H8'", "V7", "ctx_lr+fb")]
    corpora = [c for c in PATIENT if c[0] == TAG]
    assert corpora, f"{TAG} not in PATIENT"
    lines = ["# e22: frozen V7 on the LLM-patient v2 corpus (exploratory; V7 is not registered here)", ""]
    numbers = {}
    e16.block(corpora, "patient_v2", lines, numbers)
    lines.append(f"runtime {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
