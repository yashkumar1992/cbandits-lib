# core/interfaces/action.py

from __future__ import annotations
from typing import Protocol, Any, Dict


class Action(Protocol):
    """
    Minimal action abstraction for contextual bandits:

      1. get_id        -> unique arm identifier
      2. features      -> model-input feature mapping (vector or namespace dict)
      3. record_stats  -> accumulate outcome statistics
    """

    def get_id(self) -> Any:
        """Return a unique identifier for this action."""

    def features(self) -> Any:
        """
        Return the features used by the model:
          - numeric vector for Python learners
          - namespace->dict mapping for VW-ADF
        """

    def record_stats(self, reward: float, **info: Any) -> None:
        """
        Record observed reward and optional metadata.
        """
