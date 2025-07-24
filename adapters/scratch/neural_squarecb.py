import numpy as np
from typing import List, Tuple
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import NeuralLinearBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import SquareCBExploration
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action_result import ActionResult
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
import torch


class NeuralSquareCB(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, hidden_dims: List[int] = [64, 32], gamma: float = 100.0):
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.initialized = False
        
        # Use NeuralLinearBandit which works with continuous action features
        policy_learner = NeuralLinearBandit(
            feature_dim=ctx_dim,  # Context dimension
            hidden_dims=self.hidden_dims,
            training_rounds=1,
            learning_rate=0.01,
            action_representation_module=IdentityActionRepresentationModule(
                representation_dim=act_dim  # Action dimension
            ),
            exploration_module=SquareCBExploration(gamma=self.gamma)
        )
        
        # Create replay buffer
        replay_buffer = BasicReplayBuffer(capacity=1000)
        
        # Create Pearl agent
        self.agent = PearlAgent(
            policy_learner=policy_learner,
            replay_buffer=replay_buffer,
            device_id=-1
        )

    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode=False):
        if not self.initialized:
            return [0.0] * len(actions)
        
        predictions = []
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        for action in actions:
            action_tensor = torch.tensor(action, dtype=torch.float32)
            
            # Use Pearl's reward model to predict
            with torch.no_grad():
                # NeuralLinearBandit uses context and action separately
                pred = self.agent.policy_learner._reward_model(context_tensor.unsqueeze(0), action_tensor.unsqueeze(0))
                predictions.append(pred.item())
        
        return predictions
    
    def choose(self, context: np.ndarray, actions: List[np.ndarray]):
        # Get predictions from Pearl
        predictions = self.predict(context, actions)
        
        # Apply SquareCB selection logic
        max_pred = max(predictions)
        gaps = [max(max_pred - pred, 0.01) for pred in predictions]
        scores = [1.0 / (gap ** 2) for gap in gaps]
        probs = [score / sum(scores) for score in scores]
        
        # Sample based on SquareCB probabilities
        return np.random.choice(len(actions), p=probs)

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):
        # Convert to tensors
        context_tensor = torch.tensor(context, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        
        # For contextual bandits, we don't need the full reset/act/observe cycle
        # We can directly create an ActionResult and observe it
        action_result = ActionResult(
            observation=context_tensor,
            reward=reward,
            terminated=False,
            truncated=False,
            info={}
        )
        
        # Directly observe the result - Pearl contextual bandits don't require the full RL cycle
        self.agent.observe(action_result)
        self.agent.learn()
        
        self.initialized = True
        return {"status": "updated"}, True

    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for context, chosen_action, reward, chosen_prob in interactions:
            self.update(context, chosen_action, reward, chosen_prob)

    def save(self, registry_root="./artifacts/models", name="neural_squarecb", version="1.0", artifact_subpath="models"):
        model_data = {
            "ctx_dim": self.ctx_dim,
            "act_dim": self.act_dim,
            "hidden_dims": self.hidden_dims,
            "gamma": self.gamma,
            "initialized": self.initialized,
            "agent_state": self.agent.state_dict()
        }
        FSModelStore().save(name, model_data, version, registry_root, artifact_subpath)

    def load(self, registry_root="./artifacts/models", name="neural_squarecb", version="1.0", artifact_subpath="models"):
        model_data = FSModelStore().load(name, version, registry_root, artifact_subpath)
        self.ctx_dim = model_data["ctx_dim"]
        self.act_dim = model_data["act_dim"] 
        self.hidden_dims = model_data["hidden_dims"]
        self.gamma = model_data["gamma"]
        self.initialized = model_data["initialized"]
        
        # Recreate Pearl agent
        policy_learner = NeuralLinearBandit(
            feature_dim=self.ctx_dim,
            hidden_dims=self.hidden_dims,
            training_rounds=1,
            learning_rate=0.01,
            action_representation_module=IdentityActionRepresentationModule(
                representation_dim=self.act_dim
            ),
            exploration_module=SquareCBExploration(gamma=self.gamma)
        )
        
        replay_buffer = BasicReplayBuffer(capacity=1000)
        
        self.agent = PearlAgent(
            policy_learner=policy_learner,
            replay_buffer=replay_buffer,
            device_id=-1
        )
        
        # Load saved state
        self.agent.load_state_dict(model_data["agent_state"])