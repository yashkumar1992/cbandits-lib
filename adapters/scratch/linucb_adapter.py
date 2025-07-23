
import numpy as np
from core.interfaces.cb_model import CBModel
from typing import List, Tuple
from registry.registry_fs import FSModelStore

class LinUCBModel(CBModel):
    def __init__(self, ctx_dim:int, act_dim:int, alpha:float)-> None:
        """
        Initialize LinUCB model with context and action dimensions and exploration factor
        """
        self.alpha = alpha
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim

        # Initialize context weight vectors
        self.A0 = np.identity(ctx_dim)
        self.b0 = np.zeros(ctx_dim)
        self.beta = np.linalg.inv(self.A0) @ self.b0

        # Initialize action weight vectors
        self.A = np.identity(act_dim)
        self.b = np.zeros(act_dim)
        self.theta = np.linalg.inv(self.A) @ self.b

    def predict(self, context: np.ndarray, actions: List[np.ndarray], alpha:float) -> float:
        scores = []
        for action in actions:
            pred = context @ self.beta + action @ self.theta
            exploration = alpha * np.sqrt(context @ np.linalg.inv(self.A0) @ context +
                                       action @ np.linalg.inv(self.A) @ action)
            score = pred + alpha*exploration
            scores.append(score)
        return np.argmax(scores)
    
    def update(self, context: np.ndarray, action: np.ndarray, reward:float):
        self.A0 += np.outer(context, context)
        self.b0 += reward * context
        self.beta = np.linalg.inv(self.A0) @ self.b0

        self.A += np.outer(action, action)
        self.b += reward * action
        self.theta = np.linalg.inv(self.A) @ self.b

        return {"status":"updated"},True
    
    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        """
        Batch update for multiple interactions. (context, chosen action, reward, chosen_prob)
        """
        for context, chosen_action, reward, _ in interactions:
            self.update(context, chosen_action, reward)

    
    def save(self, registry_root="./artifacts/models", name = "linucb", version = "1.0", artifact_subpath = "models"):
        model_data = {
            "alpha": self.alpha,
            "ctx_dim": self.ctx_dim,
            "act_dim": self.act_dim,
            "A0": self.A0,
            "b0": self.b0,
            "beta": self.beta,
            "A": self.A,
            "b": self.b,
            "theta": self.theta,
        }
        
        model_store = FSModelStore()
        model_store.save(
            name=name,
            model_data=model_data,
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath
        )
        

    def load(self, registry_root="./artifacts/models", name = "linucb",version = "1.0", artifact_subpath = "models"):
        model_store = FSModelStore()
        model_data = model_store.load(
            name= name,
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath
        )
        
        self.alpha = model_data["alpha"]
        self.ctx_dim = model_data["ctx_dim"]
        self.act_dim = model_data["act_dim"]
        self.A0 = model_data["A0"]
        self.b0 = model_data["b0"]
        self.beta = model_data["beta"]
        self.A = model_data["A"]
        self.b = model_data["b"]
        self.theta = model_data["theta"]


