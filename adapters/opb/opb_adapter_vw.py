# adapters/opb/opb_adapter_vw.py
import yaml
import numpy as np
from typing import Any, Sequence, Tuple, Optional, Dict, List
from core.interfaces.cb_model import CBModel
from core.interfaces.action import Action
from core.utils.vw_parser import VWFormatParser

# OBP imports
from obp.policy import (
    LinUCB, LogisticUCB, Random, EpsilonGreedy, 
    LinTS, LogisticTS, LinEpsilonGreedy
)
import json

class SimpleActionVW:
    """Simple action implementation for VW format data"""
    def __init__(self, action_id: int, features: Dict[str, Any]):
        self.action_id = action_id
        self.action_features = features
    
    def get_id(self):
        return self.action_id
    
    def features(self):
        return self.action_features
    
    def to_vector(self, feature_names: List[str]) -> np.ndarray:
        """Convert to feature vector for linear models"""
        return np.array([
            self.action_features.get(name, 0.0) if isinstance(self.action_features.get(name, 0.0), (int, float)) 
            else hash(str(self.action_features.get(name, 0.0))) % 1000 
            for name in feature_names
        ], dtype=float)

class OpenBanditsAdapterVW(CBModel):
    """Enhanced OBP adapter with VW format support and multiple algorithms"""
    
    # Algorithm registry with their required parameters
    ALGORITHMS = {
        'linear_ucb': {
            'class': LinUCB,
            'contextual': True,
            'params': ['dim', 'n_actions', 'random_state']
        },
        'logistic_ucb': {
            'class': LogisticUCB, 
            'contextual': True,
            'params': ['dim', 'n_actions', 'random_state']
        },
        'linear_ts': {
            'class': LinTS,
            'contextual': True,
            'params': ['dim', 'n_actions', 'random_state']
        },
        'logistic_ts': {
            'class': LogisticTS,
            'contextual': True,
            'params': ['dim', 'n_actions', 'random_state']
        },
        'linear_epsilon_greedy': {
            'class': LinEpsilonGreedy,
            'contextual': True,
            'params': ['dim', 'n_actions', 'epsilon', 'random_state']
        },
        'epsilon_greedy': {
            'class': EpsilonGreedy,
            'contextual': False,
            'params': ['n_actions', 'epsilon', 'random_state']
        },
        'random': {
            'class': Random,
            'contextual': False,
            'params': ['n_actions', 'random_state']
        }
    }
    
    def __init__(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        ob_cfg = cfg.get('open_bandits_adapter', {})
        algorithm = ob_cfg['algorithm']
        params = ob_cfg.get('params', {})
        
        if algorithm not in self.ALGORITHMS:
            available = list(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        
        # Initialize the policy
        self.algorithm_name = algorithm
        self.algorithm_config = self.ALGORITHMS[algorithm]
        self.policy_class = self.algorithm_config['class']
        self.params = params
        self.is_contextual = self.algorithm_config['contextual']
        
        # Will be set when we know the dimensions
        self.policy = None
        self.n_actions = None
        self.context_dim = None
        self.context_feature_names = []
        self.action_feature_names = []
        
        # VW parser
        self.vw_parser = VWFormatParser()
        
        # Store interactions for batch learning
        self.interactions = []

    def _initialize_policy(self, n_actions: int, context_dim: int):
        """Initialize policy with known dimensions"""
        if self.policy is None:
            self.n_actions = n_actions
            self.context_dim = context_dim
            
            # Build initialization parameters
            init_params = {
                'n_actions': n_actions,
                'random_state': self.params.get('random_state', 12345)
            }
            
            # Add contextual parameters if needed
            if self.is_contextual:
                init_params['dim'] = context_dim
            
            # Add algorithm-specific parameters
            if 'epsilon' in self.algorithm_config['params']:
                init_params['epsilon'] = self.params.get('epsilon', 0.1)
            
            # Add any other parameters from config
            for param_name, param_value in self.params.items():
                if param_name not in init_params and param_name in self.algorithm_config['params']:
                    init_params[param_name] = param_value
            
            # Initialize the policy
            self.policy = self.policy_class(**init_params)
            
            print(f"✅ Initialized {self.algorithm_name} with params: {init_params}")

    def _convert_to_vector(self, features: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
        """Convert feature dict to vector using consistent feature ordering"""
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            if isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                # Hash string features to numeric values
                vector.append(float(hash(str(value)) % 1000))
        return np.array(vector, dtype=float)

    def get_exploration_strategy(self) -> Any:
        return {
            'algorithm': self.algorithm_name,
            'params': self.params,
            'contextual': self.is_contextual
        }

    def predict(
        self,
        row: Tuple[Any, Sequence[Action]],
        eval_mode: bool = False
    ) -> Tuple[Any, float]:
        context, actions = row
        
        # Convert context and actions to vectors
        if isinstance(context, dict):
            context_vec = self._convert_to_vector(context, self.context_feature_names)
        else:
            context_vec = np.array(context, dtype=float)
        
        # Initialize policy if needed
        if self.policy is None:
            # Determine dimensions from data
            if not self.context_feature_names and isinstance(context, dict):
                self.context_feature_names = sorted(context.keys())
            if not self.action_feature_names and hasattr(actions[0], 'features'):
                self.action_feature_names = sorted(actions[0].features().keys())
            
            context_dim = len(self.context_feature_names) if self.context_feature_names else len(context_vec)
            self._initialize_policy(len(actions), context_dim)
        
        # For contextual algorithms, use context in prediction
        if self.is_contextual and hasattr(self.policy, 'predict'):
            try:
                action_indices = self.policy.predict(context=context_vec.reshape(1, -1))
                chosen_action = action_indices[0] if isinstance(action_indices, (list, np.ndarray)) else action_indices
            except Exception as e:
                print(f"Warning: Policy prediction failed: {e}. Using random fallback.")
                chosen_action = np.random.randint(len(actions))
        else:
            # For non-contextual algorithms or fallback
            chosen_action = np.random.randint(len(actions))
        
        # Return action ID and uniform probability (OBP handles exploration internally)
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
            if data_file.endswith('.dat'):
                self._learn_from_vw_file(data_file)
            else:
                self._learn_from_json_file(data_file)

    def _learn_from_vw_file(self, data_file: str):
        """Learn from VW format file"""
        print(f"📖 Parsing VW format file: {data_file}")
        interactions = self.vw_parser.parse_vw_file(data_file)
        
        if not interactions:
            raise ValueError(f"No interactions found in {data_file}")
        
        # Extract feature names for consistent vectorization
        all_context_features = set()
        all_action_features = set()
        
        for context, actions, _ in interactions:
            all_context_features.update(context.keys())
            for action in actions:
                all_action_features.update(action.keys())
        
        self.context_feature_names = sorted(list(all_context_features))
        self.action_feature_names = sorted(list(all_action_features))
        
        print(f"📊 Found {len(interactions)} interactions")
        print(f"📊 Context features: {self.context_feature_names}")
        print(f"📊 Action features: {self.action_feature_names}")
        
        # Initialize policy with determined dimensions
        context_dim = len(self.context_feature_names)
        n_actions = len(interactions[0][1]) if interactions else 3  # fallback
        self._initialize_policy(n_actions, context_dim)
        
        # Store interactions for replay learning
        for context, actions, label_info in interactions:
            # Convert to our internal format
            action_objects = [
                SimpleActionVW(i, action_dict) 
                for i, action_dict in enumerate(actions)
            ]
            
            if label_info:
                chosen_idx, reward, prob = label_info
                self.update((context, chosen_idx, reward, prob))

    def _learn_from_json_file(self, data_file: str):
        """Learn from JSON lines file (fallback)"""
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                context = data['context']
                actions = data['actions']
                chosen_idx = data['chosen_action']
                reward = data['reward']
                prob = data.get('probability', 1.0)
                
                # Initialize policy if needed
                if self.policy is None:
                    if isinstance(context, dict):
                        context_dim = len(context)
                    else:
                        context_dim = len(context) if hasattr(context, '__len__') else 1
                    self._initialize_policy(len(actions), context_dim)
                
                self.update((context, chosen_idx, reward, prob))

    def reset(self) -> None:
        self.policy = None
        self.interactions = []
        self.context_feature_names = []
        self.action_feature_names = []

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
            'algorithm_config': self.algorithm_config,
            'params': self.params,
            'n_actions': self.n_actions,
            'context_dim': self.context_dim,
            'context_feature_names': self.context_feature_names,
            'action_feature_names': self.action_feature_names,
            'interactions': self.interactions,
            'is_contextual': self.is_contextual
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
    ) -> 'OpenBanditsAdapterVW':
        import os
        import joblib
        
        filepath = os.path.join(registry_root, name, version, artifact_subpath, f"{name}-{version}.joblib")
        save_data = joblib.load(filepath)
        
        # Create instance
        inst = cls.__new__(cls)
        inst.policy = save_data['policy']
        inst.algorithm_name = save_data['algorithm_name']
        inst.algorithm_config = save_data['algorithm_config']
        inst.params = save_data['params']
        inst.n_actions = save_data['n_actions']
        inst.context_dim = save_data['context_dim']
        inst.context_feature_names = save_data['context_feature_names']
        inst.action_feature_names = save_data['action_feature_names']
        inst.interactions = save_data['interactions']
        inst.is_contextual = save_data['is_contextual']
        inst.vw_parser = VWFormatParser()
        
        return inst