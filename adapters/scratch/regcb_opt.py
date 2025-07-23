
import numpy as np
from typing import List, Tuple
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore


class RegCBOpt(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, alpha: float = 1.0):
        self.d = ctx_dim + act_dim
        self.alpha = alpha
        self.A = np.identity(self.d)
        self.b = np.zeros(self.d)
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b
        self.initialized = False

    def _features(self, context, action):
        return np.concatenate([context, action])

    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode=False):
        if not self.initialized:
            return [0.0] * len(actions)

        preds = []
        for action in actions:
            x = self._features(context, action)
            mean = x @ self.theta
            delta = self.alpha * np.sqrt(x @ self.A_inv @ x)
            preds.append(mean + delta)
        return preds

    def choose(self, context: np.ndarray, actions: List[np.ndarray]):
        preds = self.predict(context, actions)
        return int(np.argmax(preds))

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):
        x = self._features(context, action)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b
        self.initialized = True
        return {"status": "updated"}, True

    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for context, action, reward, _ in interactions:
            self.update(context, action, reward)

    def save(self, registry_root="./artifacts/models", name="regcb_opt", version="1.0", artifact_subpath="models"):
        model_data = {
            "A": self.A,
            "b": self.b,
            "A_inv": self.A_inv,
            "theta": self.theta,
            "alpha": self.alpha,
            "initialized": self.initialized
        }
        FSModelStore().save(name, model_data, version, registry_root, artifact_subpath)

    def load(self, registry_root="./artifacts/models", name="regcb_opt", version="1.0", artifact_subpath="models"):
        model_data = FSModelStore().load(name, version, registry_root, artifact_subpath)
        self.A = model_data["A"]
        self.b = model_data["b"]
        self.A_inv = model_data["A_inv"]
        self.theta = model_data["theta"]
        self.alpha = model_data["alpha"]
        self.initialized = model_data["initialized"]
