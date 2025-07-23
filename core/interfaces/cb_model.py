# core/interfaces/cb_model.py

from __future__ import annotations
from typing import Protocol, Any, Sequence, Tuple, Optional


class CBModel(Protocol):
    """
    Contextual-bandit model interface using tuple-based rows:
      - `predict`       : (context, candidate_actions) -> (action_id, score)
      - `batch_predict` : list of (context, candidate_actions) -> list of (action_id, score)
      - `update`        : (context, action_id, reward, prob_logged) -> None
      - `batch_update`  : list of interactions for batch updates
      - `get_exploration_strategy`: inspect exploration settings
      - `reset`, `save`, `load`: lifecycle management
    """

    def __init__(self, **model_params: Any) -> None:
        """
        Initialize model with arbitrary parameters:
        e.g. for VW: vw_args="--cb_explore_adf --epsilon 0.2"
             for LinUCB: dim=3, alpha=1.0, lmbda=0.5, etc.
        """

    def get_exploration_strategy(self) -> Any:
        """Return exploration strategy info (e.g. VW args or epsilon)."""

    def predict(
        self,
        eval_mode: bool = False,
        *kwargs: Any
    ) -> Any:
        """
        Single-row inference.

        :param row: (context, candidate_actions)
        :param eval_mode: if True, disable exploration
        :returns: (chosen_action_id, score)
        """

    def choose(
        self,
        *kwargs: Any
    ) -> Any:
        """
        Single-row choose.
        :returns: (chosen_action_id, score)
        """

    def batch_predict(
        self,
        eval_mode: bool = False,
        *kwargs: Any
    ) -> Any:
        """
        Batch inference over multiple rows.
        """

    def update(
        self,
        *kwargs: Any
    ) -> None:
        """
        Single-step online update.

        :param interaction: (context, action_id, reward, prob_logged)
        """

    def batch_update(
        self,
        data_file: Optional[str] = None,
        *kwargs: Any
    ) -> None:
        """
        Batch-style update over many interactions.
        """

    def reset(self) -> None:
        """Reinitialize internal state."""

    def save(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> str:
        """
        Persist model state; return URI or path.
        """

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> CBModel:
        """
        Reconstruct a `CBModel` from saved artifact.
        """
