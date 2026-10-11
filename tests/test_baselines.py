"""Tests for the baseline detectors (semantic entropy, SelfCheckGPT, CoVe,
GCN). All LLM dependencies are fakes — no API calls anywhere."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.bootstrap_ci import auroc
from pge.baselines.cove import ChainOfVerification
from pge.baselines.gcn import GCNClassifier
from pge.baselines.selfcheckgpt import SelfCheckGPT
from pge.baselines.semantic_entropy import (
    MeanTurnEntropy,
    SemanticEntropy,
    _normalize,
)
from pge.dag import ChainOfThoughtDAG, GuessNode
from pge.trajectory import Trajectory, Turn

# --------------------------------------------------------------- fixtures


def sample_text(value: str, p: float = 0.8) -> str:
    return f"THOUGHT: reconsidering.\n[final_answer]: {value}, p={p}"


def make_turn(
    attribute="attr",
    guess="x",
    prob=0.8,
    turn=0,
    samples=(),
    prompt="",
    reasoning="",
) -> Turn:
    return Turn(
        attribute=attribute,
        guess_value=guess,
        probability=prob,
        turn=turn,
        prompt=prompt,
        reasoning=reasoning,
        samples=list(samples),
    )


def make_traj(turns, task_id="t0") -> Trajectory:
    return Trajectory(
        task_id=task_id, domain="test", model="fake", seed=0, turns=turns
    )


# --------------------------------------------------------- semantic entropy


def test_normalize_numeric_and_text():
    assert _normalize("2") == _normalize("2.0") == _normalize(" 2.00 ")
    assert _normalize("Paris.") == _normalize("  paris") == "paris"
    assert _normalize('"New   York"') == "new york"


def test_entropy_zero_for_unanimous_samples():
    t = make_turn(guess="42", samples=[sample_text("42")] * 4)
    assert SemanticEntropy().score(make_traj([t])) == 0.0


def test_entropy_positive_for_split_samples():
    t = make_turn(
        guess="paris",
        samples=[sample_text("london"), sample_text("london"), sample_text("paris")],
    )
    # answers: paris, london, london, paris -> 50/50 split -> ln 2
    s = SemanticEntropy().score(make_traj([t]))
    assert s == pytest.approx(math.log(2))


def test_entropy_numeric_equivalence():
    t = make_turn(guess="2", samples=[sample_text("2.0"), sample_text("2.00")])
    assert SemanticEntropy().score(make_traj([t])) == 0.0


def test_entropy_no_samples_is_zero():
    t = make_turn(guess="7", samples=[])
    assert SemanticEntropy().score(make_traj([t])) == 0.0
    assert SemanticEntropy().score(make_traj([])) == 0.0


def test_entropy_scores_final_turn_only():
    noisy = make_turn(
        guess="a", turn=0, samples=[sample_text("a"), sample_text("b")]
    )
    clean = make_turn(guess="c", turn=1, samples=[sample_text("c")] * 3)
    assert SemanticEntropy().score(make_traj([noisy, clean])) == 0.0


def test_llm_cluster_requires_entail_fn():
    with pytest.raises(ValueError):
        SemanticEntropy(cluster="llm")
    with pytest.raises(ValueError):
        SemanticEntropy(cluster="bogus")


def test_llm_cluster_merges_paraphrases():
    synonyms = {"paris", "the city of paris"}

    def entail_fn(a: str, b: str) -> bool:
        a, b = _normalize(a), _normalize(b)
        return a == b or (a in synonyms and b in synonyms)

    t = make_turn(
        guess="Paris",
        samples=[sample_text("the city of paris"), sample_text("London")],
    )
    s = SemanticEntropy(cluster="llm", entail_fn=entail_fn).score(make_traj([t]))
    # clusters: {paris, the city of paris} (2), {london} (1)
    expected = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert s == pytest.approx(expected)
    # exact clustering would see 3 distinct answers instead
    s_exact = SemanticEntropy().score(make_traj([t]))
    assert s_exact == pytest.approx(math.log(3))


def test_mean_turn_entropy_averages_turns_with_samples():
    unanimous = make_turn(guess="1", turn=0, samples=[sample_text("1")] * 2)
    split = make_turn(
        guess="a", turn=1, samples=[sample_text("b"), sample_text("b"), sample_text("a")]
    )
    no_samples = make_turn(guess="z", turn=2, samples=[])
    traj = make_traj([unanimous, split, no_samples])
    # entropies: 0 and ln 2; turn without samples excluded
    assert MeanTurnEntropy().score(traj) == pytest.approx(math.log(2) / 2)
    assert MeanTurnEntropy().score(make_traj([no_samples])) == 0.0


# ------------------------------------------------------------ selfcheckgpt


def test_selfcheck_fraction_correctness():
    t = make_turn(
        guess="42",
        samples=[
            sample_text("42"),
            sample_text("42.0"),  # numeric match
            sample_text("41"),
            sample_text("43"),
        ],
    )
    assert SelfCheckGPT().score(make_traj([t])) == pytest.approx(0.5)


def test_selfcheck_mean_over_turns():
    t0 = make_turn(turn=0, guess="a", samples=[sample_text("a")] * 2)  # 0.0
    t1 = make_turn(
        turn=1, guess="b", samples=[sample_text("c"), sample_text("c")]
    )  # 1.0
    t2 = make_turn(turn=2, guess="d", samples=[])  # excluded
    assert SelfCheckGPT().score(make_traj([t0, t1, t2])) == pytest.approx(0.5)


def test_selfcheck_no_samples_zero():
    assert SelfCheckGPT().score(make_traj([make_turn()])) == 0.0
    assert SelfCheckGPT().score(make_traj([])) == 0.0


def test_selfcheck_unparseable_sample_counts_as_unsupported():
    t = make_turn(guess="42", samples=["no commitment line here", sample_text("42")])
    assert SelfCheckGPT().score(make_traj([t])) == pytest.approx(0.5)


def test_selfcheck_injectable_check_fn():
    calls = []

    def check_fn(claim: str, sample: str) -> bool:
        calls.append((claim, sample))
        return claim in sample

    t = make_turn(
        guess="42", samples=["the answer 42 appears here", "nothing relevant"]
    )
    s = SelfCheckGPT(check_fn=check_fn).score(make_traj([t]))
    assert s == pytest.approx(0.5)
    assert len(calls) == 2 and calls[0][0] == "42"


# --------------------------------------------------------------------- cove


def test_cove_requires_llm_fn():
    with pytest.raises(ValueError):
        ChainOfVerification(llm_fn=None)
    with pytest.raises(ValueError):
        ChainOfVerification()


def test_cove_disagreement_fraction_and_calls():
    def fake_llm(prompt: str) -> str:
        if '"color"' in prompt:
            return "[final_answer]: blue, p=0.9"
        return "[final_answer]: 7, p=0.9"

    turns = [
        make_turn(attribute="color", guess="Blue", turn=0, prompt="what color?"),
        make_turn(attribute="size", guess="7.0", turn=1, prompt="what size?"),
        make_turn(attribute="shape", guess="round", turn=2, prompt="what shape?"),
    ]
    cove = ChainOfVerification(llm_fn=fake_llm)
    # agree (case), agree (numeric 7 vs 7.0), disagree (7 vs round)
    assert cove.score(make_traj(turns)) == pytest.approx(1 / 3)
    assert cove.calls_made == 3
    cove.score(make_traj(turns))
    assert cove.calls_made == 6


def test_cove_uses_prompt_or_falls_back_to_reasoning():
    seen = []

    def fake_llm(prompt: str) -> str:
        seen.append(prompt)
        return "[final_answer]: x, p=0.5"

    turns = [
        make_turn(attribute="a", guess="x", turn=0, prompt="ORIGINAL PROMPT"),
        make_turn(
            attribute="b", guess="x", turn=1, prompt="", reasoning="TRAJ REASONING"
        ),
    ]
    ChainOfVerification(llm_fn=fake_llm).score(make_traj(turns))
    assert "ORIGINAL PROMPT" in seen[0]
    assert "TRAJ REASONING" in seen[1]


def test_cove_all_agree_scores_zero():
    def fake_llm(prompt: str) -> str:
        return "[final_answer]: yes, p=0.9"

    turns = [make_turn(attribute="a", guess="yes", turn=i) for i in range(3)]
    assert ChainOfVerification(llm_fn=fake_llm).score(make_traj(turns)) == 0.0


# ---------------------------------------------------------------------- gcn


def _chain_dag(rng, n_nodes: int, hallucinated: bool) -> ChainOfThoughtDAG:
    G = ChainOfThoughtDAG()
    for i in range(n_nodes):
        if hallucinated:
            score = -0.6 + rng.normal(0, 0.15)
            prob = 0.85 + rng.normal(0, 0.05)
        else:
            score = 0.6 + rng.normal(0, 0.15)
            prob = 0.7 + rng.normal(0, 0.05)
        G.add_node(
            GuessNode(
                attribute=f"a{i}",
                value=i,
                probability=float(np.clip(prob, 0, 1)),
                score=float(np.clip(score, -1, 1)),
                turn=i,
            )
        )
        if i > 0:
            G.add_edge(i - 1, i)
    return G


def _synthetic_dataset(n=60, seed=7):
    rng = np.random.default_rng(seed)
    dags, labels = [], []
    for i in range(n):
        y = i % 2
        dags.append(_chain_dag(rng, int(rng.integers(3, 9)), hallucinated=bool(y)))
        labels.append(y)
    trajs = [make_traj([make_turn()], task_id=f"g{i}") for i in range(n)]
    return trajs, dags, labels


def test_gcn_requires_dags_and_labels():
    with pytest.raises(ValueError):
        GCNClassifier().fit([], dags=None, labels=None)


def test_gcn_requires_fit_before_score():
    with pytest.raises(RuntimeError):
        GCNClassifier().score(make_traj([make_turn()]), dag=ChainOfThoughtDAG())


def test_gcn_requires_dag_at_score_time():
    trajs, dags, labels = _synthetic_dataset(n=10)
    clf = GCNClassifier(epochs=5).fit(trajs, dags=dags, labels=labels)
    with pytest.raises(ValueError):
        clf.score(trajs[0], dag=None)


def test_gcn_separates_easy_synthetic_dataset():
    trajs, dags, labels = _synthetic_dataset(n=60)
    clf = GCNClassifier().fit(trajs, dags=dags, labels=labels)
    scores = np.array(
        [clf.score(tr, dag=G) for tr, G in zip(trajs, dags)]
    )
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert auroc(scores, np.array(labels)) > 0.8


def test_gcn_handles_variable_and_empty_graphs():
    trajs, dags, labels = _synthetic_dataset(n=20)
    clf = GCNClassifier(epochs=20).fit(trajs, dags=dags, labels=labels)
    s = clf.score(make_traj([make_turn()]), dag=ChainOfThoughtDAG())
    assert 0.0 <= s <= 1.0
