"""Agent interface for the PGE loop (paper Section 3.1).

An Agent reads the user's setup answers and the partial chain-of-thought, and
emits a calibrated probabilistic guess at each turn:

    "[Attribute]: [value], p=[probability]"

The Protocol consumes these guesses and feeds them to the scoring + graphon
pipeline. We define an ABC so a real LLM-backed agent can be dropped in
without touching the rest of the framework; `StubAgent` provides a
deterministic rule-based agent for the demo and tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from pge.dag import ChainOfThoughtDAG


@dataclass
class Guess:
    attribute: str
    value: object
    probability: float


class Agent(ABC):
    """Abstract interface; replace with an LLM-backed implementation in production."""

    @abstractmethod
    def guess(
        self,
        turn: int,
        setup_answers: Mapping[str, object],
        history: ChainOfThoughtDAG,
    ) -> Guess:
        """Produce one calibrated guess for the current turn."""


class StubAgent(Agent):
    """Rule-based agent for the demo.

    Maintains a fixed schedule of attributes to guess. For each (turn, attribute)
    it derives the most-likely value from the setup answers using a small
    lookup, then emits a calibrated probability that is high when the setup
    answer is informative and low when it isn't.

    Parameters
    ----------
    attribute_schedule : list[str]
        Order in which attributes are guessed across turns.
    value_table : dict[str, dict[answer_key, (value, probability)]]
        For each attribute, mapping from a *key* (a tuple of relevant setup-answer
        values) to the (guessed_value, probability) pair the agent should output.
    relevant_keys : dict[str, tuple[str, ...]]
        For each attribute, which setup-answer fields go into the lookup key.
    noise : float
        If > 0, gaussian noise with std `noise` is added to the probability
        (clipped to (0, 1)). Used to inject calibrated uncertainty.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        attribute_schedule: list[str],
        value_table: dict,
        relevant_keys: dict,
        noise: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.attribute_schedule = attribute_schedule
        self.value_table = value_table
        self.relevant_keys = relevant_keys
        self.noise = float(noise)
        self.rng = rng if rng is not None else np.random.default_rng()

    def guess(self, turn, setup_answers, history):
        attribute = self.attribute_schedule[turn % len(self.attribute_schedule)]
        keys = self.relevant_keys[attribute]
        lookup = tuple(setup_answers.get(k) for k in keys)
        value, p = self.value_table[attribute].get(
            lookup,
            # Fallback: uniform-ish guess
            (self.value_table[attribute].get("__default__", ("unknown", 0.5)))[0]
            if "__default__" in self.value_table[attribute]
            else ("unknown", 0.5),
        )
        if self.noise > 0:
            p = float(np.clip(p + self.rng.normal(0.0, self.noise), 0.01, 0.99))
        return Guess(attribute=attribute, value=value, probability=float(p))
