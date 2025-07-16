from typing import Any, Tuple, Dict, Optional
import numpy as np

class CBEnv:
    """
    A toy contextual-bandit environment with 3-dimensional features.
    Reward = dot(weight_vector, action_features).
    """

    def __init__(self, weights: np.ndarray = np.array([1.0, 0.5, -0.2])):
        if weights.shape != (3,):
            raise ValueError("Weights must be a 1D array of length 3.")
        self.weights = weights

    def reset(self) -> None:
        """
        No state to reset in a contextual bandit.
        """
        return None

    def step(
        self, 
        action_features: np.ndarray
    ) -> Tuple[Optional[Any], float, bool, Dict]:
        """
        Simulate taking an action described by its feature vector.

        :param action_features: np.ndarray of shape (3,)
        :returns: (next_state, reward, done, info)
        """
        if action_features.shape != (3,):
            raise ValueError("Action features must be a 1D array of length 3.")
        
        reward = float(self.weights.dot(action_features))
        done = False     # each pull is independent
        info = {}
        return None, reward, done, info
