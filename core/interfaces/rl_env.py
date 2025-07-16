# core/interfaces/rl_env.py
from typing import Protocol, Any, Tuple, Dict

class RLEnv(Protocol):
    """
    Generic reinforcement-learning environment (Gym-style).
    """
    def reset(self) -> Any: ...

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, Dict]:
        """
        Returns
        -------
        next_state : Any
        reward     : float
        done       : bool
        info       : dict          (diagnostics; may be empty)
        """
