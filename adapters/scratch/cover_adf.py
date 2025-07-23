
import numpy as np
from typing import List, Tuple
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore


class CoverADF(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, cover_size: int = 8, epsilon: float = 0.05, psi: float = 1.0):
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.cover_size = cover_size
        self.epsilon = epsilon
        self.psi = psi

        self.policies = [
            {"w_ctx": np.zeros(ctx_dim), "w_act": np.zeros(act_dim)}
            for _ in range(cover_size)
        ]
        self.initialized = False

    def _score(self, context, action, policy):
        return context @ policy["w_ctx"] + action @ policy["w_act"]

    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode=False):
        if not self.initialized:
            return [1.0 / len(actions)] * len(actions)

        cover_scores = np.zeros(len(actions))
        for policy in self.policies:
            for i, action in enumerate(actions):
                cover_scores[i] += self._score(context, action, policy)

        cover_scores /= self.cover_size
        smoothed = (1 - self.epsilon) * cover_scores + self.epsilon / len(actions)
        return smoothed.tolist()

    def choose(self, context: np.ndarray, actions: List[np.ndarray]):
        probs = self.predict(context, actions)
        return int(np.argmax(probs))

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):
        self.initialized = True
        for i, policy in enumerate(self.policies):
            grad_ctx = (reward - context @ policy["w_ctx"] - action @ policy["w_act"]) * context
            grad_act = (reward - context @ policy["w_ctx"] - action @ policy["w_act"]) * action
            policy["w_ctx"] += 0.1 * grad_ctx
            policy["w_act"] += 0.1 * grad_act
        return {"status": "updated"}, True

    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for context, action, reward, _ in interactions:
            self.update(context, action, reward)

    def save(self, registry_root="./artifacts/models", name="cover_adf", version="1.0", artifact_subpath="models"):
        model_data = {
            "ctx_dim": self.ctx_dim,
            "act_dim": self.act_dim,
            "cover_size": self.cover_size,
            "epsilon": self.epsilon,
            "psi": self.psi,
            "policies": self.policies
        }
        FSModelStore().save(name, model_data, version, registry_root, artifact_subpath)

    def load(self, registry_root="./artifacts/models", name="cover_adf", version="1.0", artifact_subpath="models"):
        model_data = FSModelStore().load(name, version, registry_root, artifact_subpath)
        self.ctx_dim = model_data["ctx_dim"]
        self.act_dim = model_data["act_dim"]
        self.cover_size = model_data["cover_size"]
        self.epsilon = model_data["epsilon"]
        self.psi = model_data["psi"]
        self.policies = model_data["policies"]
        self.initialized = True
