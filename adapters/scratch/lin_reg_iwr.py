import numpy as np
from typing import List, Tuple
from sklearn.linear_model import SGDRegressor
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore

from typing import Optional



class LinRegIWR(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, learning_rate: float, init_weights: Optional[List[float]] = None):
        self.n_features = ctx_dim + act_dim
        self._model = SGDRegressor(
            max_iter=1,
            warm_start=True,
            learning_rate='constant',
            eta0=learning_rate
        )
        self.initialized = False

        if init_weights is not None:
            assert len(init_weights) == self.n_features, f"Expected {self.n_features} weights, got {len(init_weights)}"

            # Dummy fit to initialize internal structures
            X_dummy = np.zeros((1, self.n_features))
            y_dummy = np.zeros(1)
            self._model.partial_fit(X_dummy, y_dummy)

            self._model.coef_ = np.array(init_weights)
            self._model.intercept_ = np.array([0.0])  # Optional: adjust if needed
            self.initialized = True


    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode = False):
        if not self.initialized:
            return [0.0] * len(actions)
        predictions = []
        for action in actions:
            features = np.concatenate([context, action]).reshape(1, -1)
            pred = self._model.predict(features)
            predictions.append(pred)
        
        return predictions
    
    def choose(self , context:np.ndarray, actions: List[np.ndarray]):
        prediction = self.predict(context, actions)
        return np.argmax(prediction)

    def update(self, context: np.ndarray, action: np.ndarray, reward:float, chosen_prob: float = 1.0):
        x = np.concatenate([context, action]).reshape(1, -1)
        y = np.array([reward])
        weight = np.array([1/max(chosen_prob,1e-3)]) # clipping to avoid huge weights
        if not self.initialized:
            self._model.partial_fit(x, y, sample_weight=weight)
            self.initialized = True
        else:
            # Continue training with new data
            self._model.partial_fit(x,y, sample_weight=weight)

        return {"status":"updated"},True
    
    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        """
        Batch update for multiple interactions. (context, chosen action, reward, chosen_prob)
        """
        for context, chosen_action, reward, chosen_prob in interactions:
            self.update(context, chosen_action, reward, chosen_prob)

    
    def save(self, registry_root="./artifacts/models", name = "lr_iwr", version = "1.0", artifact_subpath = "models"):
        model_store = FSModelStore()
        model_store.save(
            name=name,
            model_data=self._model,
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath
        )
    
    def load(self, registry_root="./artifacts/models", name = "lr_iwr",version = "1.0", artifact_subpath = "models"):
        model_store = FSModelStore()
        model_data = model_store.load(
            name= name,
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath
        )
        self._model = model_data
        self.initialized = True

