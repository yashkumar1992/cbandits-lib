
import numpy as np
from typing import List, Tuple
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore
import random


class BaggingADF(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, num_models: int = 5, learning_rate: float = 0.1):
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.models = [
            {"w_ctx": np.zeros(ctx_dim), "w_act": np.zeros(act_dim)}
            for _ in range(num_models)
        ]
        self.initialized = False

    def _score(self, context, action, model):
        return context @ model["w_ctx"] + action @ model["w_act"]

    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode=False):
        if not self.initialized:
            return [1.0 / len(actions)] * len(actions)

        scores = np.zeros(len(actions))
        for model in self.models:
            for i, action in enumerate(actions):
                scores[i] += self._score(context, action, model)

        scores /= self.num_models
        return scores.tolist()

    def choose(self, context: np.ndarray, actions: List[np.ndarray]):
        predictions = self.predict(context, actions)
        return int(np.argmax(predictions))

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):
        self.initialized = True
        for model in self.models:
            # Bootstrap sample: randomly decide whether to train this model
            if random.random() < 0.5:
                pred = self._score(context, action, model)
                error = reward - pred
                model["w_ctx"] += self.learning_rate * error * context
                model["w_act"] += self.learning_rate * error * action
        return {"status": "updated"}, True

    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for context, action, reward, _ in interactions:
            self.update(context, action, reward)

    def save(self, registry_root="./artifacts/models", name="bagging_adf", version="1.0", artifact_subpath="models"):
        model_data = {
            "ctx_dim": self.ctx_dim,
            "act_dim": self.act_dim,
            "num_models": self.num_models,
            "learning_rate": self.learning_rate,
            "models": self.models
        }
        FSModelStore().save(name, model_data, version, registry_root, artifact_subpath)

    def load(self, registry_root="./artifacts/models", name="bagging_adf", version="1.0", artifact_subpath="models"):
        model_data = FSModelStore().load(name, version, registry_root, artifact_subpath)
        self.ctx_dim = model_data["ctx_dim"]
        self.act_dim = model_data["act_dim"]
        self.num_models = model_data["num_models"]
        self.learning_rate = model_data["learning_rate"]
        self.models = model_data["models"]
        self.initialized = True
