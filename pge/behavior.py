"""Implicit behavioral signal extraction (paper Section 3.2).

A BehaviorExtractor inspects the user's natural action at turn t (e.g.,
accept/reject a recommendation) and returns b_t in {-1, +1} indicating
whether the action is consistent with the agent's guess. No explicit
ratings are required from the user.

For the demo / tests we use a `SyntheticBehaviorExtractor` driven by a
hidden ground-truth user state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from pge.agent import Guess


class BehaviorExtractor(ABC):
    @abstractmethod
    def extract(self, guess: Guess, user_state: Mapping[str, object], turn: int) -> int:
        """Return +1 if the user's behavior is consistent with `guess`, else -1."""


class SyntheticBehaviorExtractor(BehaviorExtractor):
    """Compares each guess to a hidden ground-truth state.

    Parameters
    ----------
    truth_keys : dict[attribute, key]
        Maps each attribute name to the user_state key holding ground truth.
    """

    def __init__(self, truth_keys: dict[str, str]):
        self.truth_keys = truth_keys

    def extract(self, guess, user_state, turn):
        true_key = self.truth_keys.get(guess.attribute, guess.attribute)
        true_value = user_state.get(true_key)
        return 1 if guess.value == true_value else -1
