# adapters/opb/opb_adapter.py
import yaml
import numpy as np
from typing import Any, Sequence, Tuple, Optional, Dict, List
from core.interfaces.cb_model import CBModel
from core.interfaces.action import Action

# OBP imports
from obp.policy import LinUCB, LogisticUCB, Random, EpsilonGreedy
from obp.ope import RegressionModel
import json

class OpenBanditsAdapter(CBModel):
    """Adapter for Open Bandit Pipeline algorithms"""
    
    def __init__(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        ob_cfg = cfg.get('open_bandits_adapter', {})
        algorithm = ob_cfg['algorithm']
        params = ob_cfg.get('params', {})
        
        # Map algorithm names to OBP classes
        self.algorithm_map = {
            'linear_ucb': LinUCB,
            'logistic_ucb': LogisticUCB,
            'random': Random,
            'epsilon_greedy': EpsilonGreedy
        }
        
        if algorithm not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Initialize the policy
        self.algorithm_name = algorithm
        self.policy_class = self.algorithm_map[algorithm]
        self.params = params
        
        # Will be set when we know the dimensions
        self.policy = None
        self.n_actions = None
        self.context_dim = None
        
        # Store interactions for batch learning
        self.interactions = []

    def _initialize_policy(self, n_actions: int, context_dim: int):
        """Initialize policy with known dimensions"""
        if self.policy is None:
            self.n_actions = n_actions
            self.context_dim = context_dim
            
            if self.algorithm_name == 'linear_ucb':
                # LinUCB requires dim parameter for context dimension
                self.policy = self.policy_class(
                    dim=context_dim,
                    n_actions=n_actions,
                    random_state=self.params.get('random_state', 12345)
                )
            elif self.algorithm_name == 'logistic_ucb':
                self.policy = self.policy_class(
                    dim=context_dim,
                    n_actions=n_actions,
                    random_state=self.params.get('random_state', 12345)
                )
            elif self.algorithm_name == 'epsilon_greedy':
                self.policy = self.policy_class(
                    n_actions=n_actions,
                    epsilon=self.params.get('epsilon', 0.1),
                    random_state=self.params.get('random_state', 12345)
                )
            else:  # random
                self.policy = self.policy_class(
                    n_actions=n_actions,
                    random_state=self.params.get('random_state', 12345)
                )

    def _action_to_vector(self, action: Action) -> np.ndarray:
        """Convert action to feature vector"""
        if hasattr(action, 'to_vector'):
            return action.to_vector()
        
        features = action.features()
        if isinstance(features, dict):
            # Convert dict to vector (you may need to standardize this)
            return np.array(list(features.values()), dtype=float)
        return np.array(features, dtype=float)

    def _context_to_vector(self, context: Any) -> np.ndarray:
        """Convert context to feature vector"""
        if isinstance(context, dict):
            return np.array(list(context.values()), dtype=float)
        return np.array(context, dtype=float)

    def get_exploration_strategy(self) -> Any:
        return {
            'algorithm': self.algorithm_name,
            'params': self.params
        }

    def predict(
        self,
        row: Tuple[Any, Sequence[Action]],
        eval_mode: bool = False
    ) -> Tuple[Any, float]:
        context, actions = row
        
        # Convert to vectors
        context_vec = self._context_to_vector(context)
        action_vecs = [self._action_to_vector(a) for a in actions]
        
        # Initialize policy if needed
        if self.policy is None:
            self._initialize_policy(len(actions), len(context_vec))
        
        # Create feature matrix (context concatenated with each action)
        X = []
        for action_vec in action_vecs:
            combined = np.concatenate([context_vec, action_vec])
            X.append(combined)
        X = np.array(X)
        
        # Get action probabilities from the policy
        if hasattr(self.policy, 'predict'):
            # For contextual policies like LinUCB
            action_dist = self.policy.predict(context=context_vec)
            if isinstance(action_dist, (list, np.ndarray)):
                chosen_action = int(action_dist[0]) if len(action_dist) > 0 else 0
            else:
                chosen_action = int(action_dist)
            prob = 1.0 / len(actions)  # uniform assumption for now
        else:
            # Fallback to random
            chosen_action = np.random.randint(len(actions))
            prob = 1.0 / len(actions)
        
        return actions[chosen_action].get_id(), prob

    def batch_predict(
        self,
        rows: Sequence[Tuple[Any, Sequence[Action]]],
        eval_mode: bool = False
    ) -> Sequence[Tuple[Any, float]]:
        return [self.predict(row, eval_mode) for row in rows]

    def update(
        self,
        interaction: Tuple[Any, Any, float, Optional[float]]
    ) -> None:
        context, action_id, reward, prob = interaction
        self.interactions.append({
            'context': context,
            'action_id': action_id,
            'reward': reward,
            'probability': prob or 1.0
        })

    def batch_update(
        self,
        interactions: Optional[Sequence[Tuple[Any, Any, float, Optional[float]]]] = None,
        data_file: Optional[str] = None
    ) -> None:
        if interactions:
            for inter in interactions:
                self.update(inter)
        elif data_file:
            self._learn_from_file(data_file)

    def _learn_from_file(self, data_file: str):
        """Learn from JSON lines file"""
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                context = data['context']
                actions = data['actions']
                chosen_idx = data['chosen_action']
                reward = data['reward']
                prob = data.get('probability', 1.0)
                
                # Convert to our format
                action_objects = [
                    type('Action', (), {
                        'get_id': lambda self, i=i: i,
                        'features': lambda self, a=a: a
                    })() for i, a in enumerate(actions)
                ]
                
                # Initialize policy if needed
                if self.policy is None:
                    context_vec = self._context_to_vector(context)
                    self._initialize_policy(len(actions), len(context_vec))
                
                # Store interaction
                self.update((context, chosen_idx, reward, prob))

    def reset(self) -> None:
        self.policy = None
        self.interactions = []

    def save(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> str:
        import os
        import joblib
        
        dirpath = os.path.join(registry_root, name, version, artifact_subpath)
        os.makedirs(dirpath, exist_ok=True)
        
        save_data = {
            'policy': self.policy,
            'algorithm_name': self.algorithm_name,
            'params': self.params,
            'n_actions': self.n_actions,
            'context_dim': self.context_dim,
            'interactions': self.interactions
        }
        
        filepath = os.path.join(dirpath, f"{name}-{version}.joblib")
        joblib.dump(save_data, filepath)
        return filepath

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> 'OpenBanditsAdapter':
        import os
        import joblib
        
        filepath = os.path.join(registry_root, name, version, artifact_subpath, f"{name}-{version}.joblib")
        save_data = joblib.load(filepath)
        
        # Create instance
        inst = cls.__new__(cls)
        inst.policy = save_data['policy']
        inst.algorithm_name = save_data['algorithm_name']
        inst.params = save_data['params']
        inst.n_actions = save_data['n_actions']
        inst.context_dim = save_data['context_dim']
        inst.interactions = save_data['interactions']
        
        return inst