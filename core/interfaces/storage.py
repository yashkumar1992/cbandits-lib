# core/interfaces/storage.py

from __future__ import annotations
from typing import Protocol, Any, Mapping, Optional


class ModelStore(Protocol):
    """
    Pluggable registry back-end for saving & loading model artifacts.
    """

    def save(
        self,
        name: str,
        model: Any,
        metadata: Optional[Mapping[str, Any]] = None
    ) -> str:
        """
        Persist `model` under `name` (and metadata, e.g. hyperparams).
        Returns a `version` or URI where it was saved.
        """

    def load(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Any:
        """
        Fetch `model` by `name` and optional `version`.
        If `version=None`, load the latest.
        """

    def latest(self, name: str) -> str:
        """
        Return the latest version identifier for `name`.
        """
